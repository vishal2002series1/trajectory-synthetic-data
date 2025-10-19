"""
AWS Bedrock Provider for Claude Sonnet 4 and Titan Embeddings.
Handles text generation, vision processing, and embeddings.
"""

import json
import time
import base64
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from ..utils import get_logger

logger = get_logger(__name__)


class BedrockProvider:
    """AWS Bedrock provider for Claude and Titan models."""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-sonnet-4-20250514",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        max_tokens: int = 64000,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize Bedrock provider.
        
        Args:
            model_id: Claude model ID
            embedding_model_id: Titan embedding model ID
            region: AWS region
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
        """
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize Bedrock client
        self.client = self._initialize_client()
        logger.info(f"Initialized BedrockProvider with model: {model_id}")
    
    def _initialize_client(self):
        """Initialize AWS Bedrock client."""
        try:
            client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region
            )
            logger.debug(f"Connected to AWS Bedrock in region: {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using Claude.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop_sequences: List of stop sequences
            
        Returns:
            Generated text
        """
        # Use defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Build request body
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        if stop_sequences:
            body["stop_sequences"] = stop_sequences
        
        # Call API with retries
        response = self._call_with_retry(body)
        
        # Extract text from response
        text = self._extract_text_from_response(response)
        logger.debug(f"Generated {len(text)} characters")
        
        return text
    
    def generate_with_vision(
        self,
        prompt: str,
        image_path: Union[str, Path] = None,
        image_bytes: bytes = None,
        image_media_type: str = "image/png",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text with vision (image input).
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            image_bytes: Raw image bytes (alternative to image_path)
            image_media_type: Image MIME type
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            
        Returns:
            Generated text
        """
        # Load image
        if image_path:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            logger.debug(f"Loaded image from: {image_path}")
        elif image_bytes is None:
            raise ValueError("Either image_path or image_bytes must be provided")
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Use defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        # Call API with retries
        response = self._call_with_retry(body)
        
        # Extract text
        text = self._extract_text_from_response(response)
        logger.debug(f"Generated {len(text)} characters with vision")
        
        return text
    
    def generate_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Generate embedding using Titan.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the embedding
            
        Returns:
            Embedding vector
        """
        body = {
            "inputText": text
        }
        
        if normalize:
            body["normalize"] = True
        
        try:
            response = self.client.invoke_model(
                modelId=self.embedding_model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            logger.debug(f"Generated embedding of dimension: {len(embedding)}")
            return embedding
            
        except ClientError as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text, normalize)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(texts)} embeddings")
                    
            except Exception as e:
                logger.error(f"Error embedding text {i}: {e}")
                # Add empty embedding to maintain index alignment
                embeddings.append([])
        
        logger.info(f"Completed batch embedding: {len(embeddings)} vectors")
        return embeddings
    
    def _call_with_retry(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Bedrock API with retry logic.
        
        Args:
            body: Request body
            
        Returns:
            API response
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response['body'].read())
                return response_body
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                last_error = e
                
                # Check if we should retry
                if error_code in ['ThrottlingException', 'ServiceUnavailableException']:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"API call failed ({error_code}), "
                            f"retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                
                # Non-retryable error or max retries reached
                logger.error(f"API call failed: {error_code} - {e}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error during API call: {e}")
                raise
        
        # Max retries reached
        logger.error(f"Max retries ({self.max_retries}) reached")
        raise last_error
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract text from Bedrock response.
        
        Args:
            response: API response
            
        Returns:
            Generated text
        """
        try:
            content = response.get('content', [])
            
            if not content:
                logger.warning("No content in response")
                return ""
            
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            
            return ''.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        
        Args:
            text: Text to count
            
        Returns:
            Approximate token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BedrockProvider(model={self.model_id}, region={self.region})"


if __name__ == "__main__":
    # Test the provider
    from ..utils import setup_logger, load_config
    
    setup_logger("INFO")
    config = load_config()
    
    # Initialize provider
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region,
        max_tokens=config.bedrock.max_tokens,
        temperature=config.bedrock.temperature
    )
    
    print(f"\n✅ Provider initialized: {provider}\n")
    
    # Test text generation
    print("="*60)
    print("TEST 1: Text Generation")
    print("="*60)
    
    prompt = "What are the key benefits of using AWS Bedrock? List 3 benefits briefly."
    response = provider.generate_text(prompt)
    print(f"Response: {response}\n")
    
    # Test embedding generation
    print("="*60)
    print("TEST 2: Embedding Generation")
    print("="*60)
    
    text = "This is a test sentence for embedding."
    embedding = provider.generate_embedding(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}\n")
    
    print("="*60)
    print("✅ All tests completed!")
    print("="*60)