"""
Deduplication System using Vector Embeddings

Uses semantic similarity to detect and remove duplicate/near-duplicate
training examples using AWS Bedrock Titan embeddings.

Usage:
    from src.processing.deduplicator import Deduplicator
    
    dedup = Deduplicator(similarity_threshold=0.95)
    results = dedup.deduplicate("input.jsonl", "output.jsonl")
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

from ..core import BedrockProvider
from ..utils import get_logger, load_config

logger = get_logger(__name__)


@dataclass
class DuplicateGroup:
    """Group of duplicate examples."""
    representative_idx: int
    duplicate_indices: List[int]
    similarity_scores: List[float]
    
    def __len__(self):
        return len(self.duplicate_indices)


class Deduplicator:
    """
    Deduplicates training examples using semantic similarity.
    
    Strategy:
    1. Generate embeddings for each example (query + COT)
    2. Compute pairwise cosine similarities
    3. Cluster similar examples (similarity > threshold)
    4. Keep one representative from each cluster
    5. Remove duplicates
    
    Features:
    - Semantic deduplication (not just exact matches)
    - Configurable similarity threshold
    - Keeps the highest quality example from each cluster
    - Detailed statistics and reporting
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        batch_size: int = 20,
        config=None,
        verbose: bool = True
    ):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Cosine similarity threshold (0-1)
                                 Above this = duplicates
            batch_size: Number of examples to embed at once
            config: Configuration object (loaded automatically if None)
            verbose: If True, log detailed information
        """
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Load config and initialize provider
        self.config = config if config is not None else load_config()
        self.provider = BedrockProvider(
            model_id=self.config.bedrock.model_id,
            embedding_model_id=self.config.bedrock.embedding_model_id,
            region=self.config.bedrock.region
        )
        
        logger.info(
            f"Deduplicator initialized "
            f"(threshold={similarity_threshold}, batch_size={batch_size})"
        )
    
    def _create_embedding_text(self, example: Dict[str, Any]) -> str:
        """
        Create text for embedding from example.
        
        Combines query and COT for semantic similarity.
        """
        query = example.get("Q", "")
        cot = example.get("COT", "")
        
        # Combine query and COT
        # This captures both what's being asked and the reasoning
        text = f"Query: {query}\n\nReasoning: {cot}"
        
        return text
    
    def _generate_embeddings(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for all examples.
        
        Returns list of embedding vectors (numpy arrays).
        """
        logger.info(f"Generating embeddings for {len(examples)} examples...")
        
        embeddings = []
        total_batches = (len(examples) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(examples), self.batch_size):
            batch = examples[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            
            if self.verbose:
                logger.info(f"Processing batch {batch_num}/{total_batches}...")
            
            # Create embedding texts
            texts = [self._create_embedding_text(ex) for ex in batch]
            
            # Generate embeddings
            for text in texts:
                try:
                    embedding = self.provider.generate_embeddings(text)
                    embeddings.append(np.array(embedding))
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(1024))  # Titan embedding size
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Returns similarity score (0-1).
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure in range [0, 1]
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _find_duplicates(
        self,
        embeddings: List[np.ndarray],
        examples: List[Dict[str, Any]]
    ) -> List[DuplicateGroup]:
        """
        Find duplicate groups using embeddings.
        
        Uses greedy clustering:
        1. Start with first example as representative
        2. Find all similar examples (above threshold)
        3. Mark them as processed
        4. Move to next unprocessed example
        5. Repeat
        """
        logger.info("Finding duplicate groups...")
        
        n = len(embeddings)
        processed = set()
        duplicate_groups = []
        
        for i in range(n):
            if i in processed:
                continue
            
            # Start new group with example i as representative
            duplicates = []
            similarities = []
            
            # Compare with all other unprocessed examples
            for j in range(i + 1, n):
                if j in processed:
                    continue
                
                # Compute similarity
                similarity = self._compute_similarity(embeddings[i], embeddings[j])
                
                # If similar enough, add to duplicate group
                if similarity >= self.similarity_threshold:
                    duplicates.append(j)
                    similarities.append(similarity)
                    processed.add(j)
            
            # If found duplicates, create group
            if duplicates:
                group = DuplicateGroup(
                    representative_idx=i,
                    duplicate_indices=duplicates,
                    similarity_scores=similarities
                )
                duplicate_groups.append(group)
                
                if self.verbose:
                    logger.info(
                        f"Group {len(duplicate_groups)}: "
                        f"Example {i} has {len(duplicates)} duplicates "
                        f"(avg similarity: {np.mean(similarities):.3f})"
                    )
            
            # Mark representative as processed
            processed.add(i)
        
        logger.info(f"✅ Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups
    
    def _select_best_representative(
        self,
        examples: List[Dict[str, Any]],
        indices: List[int]
    ) -> int:
        """
        Select the best example from a duplicate group.
        
        Criteria (in order):
        1. Prefer ANSWER decisions over CALL
        2. Prefer examples with more context
        3. Prefer longer COT
        4. Use first one if all else equal
        """
        if len(indices) == 1:
            return indices[0]
        
        scores = []
        
        for idx in indices:
            example = examples[idx]
            score = 0.0
            
            # Prefer ANSWER decisions
            decision = example.get("Decision", "")
            if decision.startswith("ANSWER"):
                score += 10.0
            elif decision == "CALL":
                score += 5.0
            
            # Prefer examples with context (later iterations)
            context = example.get("Context", [])
            score += len(context) * 2.0
            
            # Prefer longer, more detailed COT
            cot = example.get("COT", "")
            score += len(cot) / 100.0
            
            scores.append((score, idx))
        
        # Return index with highest score
        best_idx = max(scores, key=lambda x: x[0])[1]
        return best_idx
    
    def deduplicate(
        self,
        input_file: str,
        output_file: str,
        stats_file: str = None,
        duplicates_file: str = None
    ) -> Dict[str, Any]:
        """
        Deduplicate training examples from input file to output file.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            stats_file: Optional path to save deduplication statistics
            duplicates_file: Optional path to save duplicate groups info
            
        Returns:
            Dictionary with deduplication statistics
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info("=" * 80)
        logger.info("DEDUPLICATION PROCESS")
        logger.info("=" * 80)
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        
        # Load examples
        logger.info("\nLoading examples...")
        examples = []
        with open(input_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        
        logger.info(f"✅ Loaded {len(examples)} examples")
        
        if len(examples) == 0:
            logger.warning("No examples to deduplicate")
            return {"total_examples": 0, "unique_examples": 0, "duplicates_removed": 0}
        
        # Generate embeddings
        embeddings = self._generate_embeddings(examples)
        
        # Find duplicate groups
        duplicate_groups = self._find_duplicates(embeddings, examples)
        
        # Determine which examples to keep
        logger.info("\nSelecting representatives...")
        indices_to_remove = set()
        duplicate_info = []
        
        for group in duplicate_groups:
            # All indices in this group
            all_indices = [group.representative_idx] + group.duplicate_indices
            
            # Select best representative
            best_idx = self._select_best_representative(examples, all_indices)
            
            # Mark others for removal
            for idx in all_indices:
                if idx != best_idx:
                    indices_to_remove.add(idx)
            
            # Save duplicate group info
            duplicate_info.append({
                "representative_idx": best_idx,
                "duplicate_indices": [i for i in all_indices if i != best_idx],
                "avg_similarity": float(np.mean(group.similarity_scores)),
                "representative_query": examples[best_idx].get("Q", "")[:100]
            })
        
        # Write unique examples
        logger.info("\nWriting unique examples...")
        unique_count = 0
        
        with open(output_path, 'w') as f:
            for idx, example in enumerate(examples):
                if idx not in indices_to_remove:
                    f.write(json.dumps(example) + "\n")
                    unique_count += 1
        
        logger.info(f"✅ Wrote {unique_count} unique examples")
        
        # Statistics
        stats = {
            "total_examples": len(examples),
            "unique_examples": unique_count,
            "duplicates_removed": len(indices_to_remove),
            "duplicate_groups": len(duplicate_groups),
            "similarity_threshold": self.similarity_threshold,
            "deduplication_rate": len(indices_to_remove) / len(examples) if examples else 0
        }
        
        # Save statistics
        if stats_file:
            stats_path = Path(stats_file)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Statistics saved to {stats_file}")
        
        # Save duplicate groups info
        if duplicates_file:
            duplicates_path = Path(duplicates_file)
            duplicates_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(duplicates_path, 'w') as f:
                json.dump(duplicate_info, f, indent=2)
            
            logger.info(f"Duplicate groups saved to {duplicates_file}")
        
        # Log summary
        logger.info("=" * 80)
        logger.info("DEDUPLICATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Unique examples: {stats['unique_examples']}")
        logger.info(f"Duplicates removed: {stats['duplicates_removed']} "
                   f"({stats['deduplication_rate']*100:.1f}%)")
        logger.info(f"Duplicate groups: {stats['duplicate_groups']}")
        
        return stats


if __name__ == "__main__":
    # Example usage
    from ..utils import setup_logger
    
    setup_logger("INFO")
    
    dedup = Deduplicator(
        similarity_threshold=0.95,
        batch_size=20,
        verbose=True
    )
    
    stats = dedup.deduplicate(
        input_file="data/output/filtered_examples.jsonl",
        output_file="data/output/final_training_data.jsonl",
        stats_file="data/output/dedup_stats.json",
        duplicates_file="data/output/duplicate_groups.json"
    )