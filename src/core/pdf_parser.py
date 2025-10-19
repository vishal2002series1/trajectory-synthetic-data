"""
Multi-modal PDF Parser for extracting text and images.
Supports vision-based image analysis with Claude.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class PDFChunk:
    """Represents a chunk of PDF content."""
    text: str
    chunk_id: int
    page_number: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


@dataclass
class PDFImage:
    """Represents an image extracted from PDF."""
    image_bytes: bytes
    page_number: int
    image_index: int
    format: str  # 'png', 'jpeg', etc.
    width: int
    height: int


class PDFParser:
    """Parse PDFs with text and image extraction."""
    
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        extract_images: bool = True,
        min_image_size: int = 100  # Minimum width/height in pixels
    ):
        """
        Initialize PDF parser.
        
        Args:
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
            extract_images: Whether to extract images
            min_image_size: Minimum image dimension to extract
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.min_image_size = min_image_size
        
        logger.info(
            f"Initialized PDFParser: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, extract_images={extract_images}"
        )
    
    def parse_pdf(
        self,
        pdf_path: str,
        vision_provider: Optional[Any] = None,
        analyze_images: bool = False,
        analyze_pages: bool = False,
        page_dpi: int = 150
    ) -> Dict[str, Any]:
        """
        Parse PDF and extract text and images.
        
        Args:
            pdf_path: Path to PDF file
            vision_provider: BedrockProvider for image analysis (optional)
            analyze_images: Whether to analyze extracted images with vision model
            analyze_pages: Whether to render and analyze each page as image (captures charts/graphs)
            page_dpi: DPI for page rendering (higher = better quality, slower)
            
        Returns:
            Dictionary with extracted content
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Parsing PDF: {pdf_path}")
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        
        # Get total pages before doing anything else
        total_pages = len(doc)
        
        # Extract text from all pages
        full_text = self._extract_text(doc)
        
        # Extract images if enabled
        images = []
        if self.extract_images:
            images = self._extract_images(doc)
            logger.info(f"Extracted {len(images)} images")
        
        # Analyze pages as images (for vector graphics, charts, etc.)
        page_descriptions = {}
        if analyze_pages and vision_provider:
            page_descriptions = self._analyze_pages_as_images(
                doc, 
                vision_provider,
                dpi=page_dpi
            )
        
        # Close document now that we're done with it
        doc.close()
        
        # Analyze extracted images with vision model if requested
        image_descriptions = {}
        if analyze_images and vision_provider and images:
            image_descriptions = self._analyze_images(images, vision_provider)
        
        # Chunk text
        chunks = self._create_chunks(
            text=full_text,
            source=pdf_path.name,
            image_descriptions=image_descriptions,
            page_descriptions=page_descriptions
        )
        
        logger.info(f"Created {len(chunks)} text chunks")
        
        return {
            'source': pdf_path.name,
            'total_pages': total_pages,
            'full_text': full_text,
            'chunks': chunks,
            'images': images,
            'image_descriptions': image_descriptions,
            'page_descriptions': page_descriptions
        }
    
    def _extract_text(self, doc: fitz.Document) -> str:
        """Extract text from all pages."""
        texts = []
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                texts.append(text)
                logger.debug(f"Extracted text from page {page_num}: {len(text)} chars")
        
        full_text = "\n\n".join(texts)
        logger.info(f"Total extracted text: {len(full_text)} characters")
        
        return full_text
    
    def _extract_images(self, doc: fitz.Document) -> List[PDFImage]:
        """Extract images from PDF."""
        images = []
        
        for page_num, page in enumerate(doc, 1):
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Get image dimensions
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size
                    
                    # Skip small images (likely icons, logos)
                    if width < self.min_image_size or height < self.min_image_size:
                        logger.debug(
                            f"Skipping small image on page {page_num}: "
                            f"{width}x{height}"
                        )
                        continue
                    
                    pdf_image = PDFImage(
                        image_bytes=image_bytes,
                        page_number=page_num,
                        image_index=img_index,
                        format=image_ext,
                        width=width,
                        height=height
                    )
                    
                    images.append(pdf_image)
                    logger.debug(
                        f"Extracted image from page {page_num}: "
                        f"{width}x{height} {image_ext}"
                    )
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_index} from page {page_num}: {e}"
                    )
        
        return images
    
    def _analyze_images(
        self,
        images: List[PDFImage],
        vision_provider: Any
    ) -> Dict[int, str]:
        """
        Analyze images with vision model.
        
        Args:
            images: List of extracted images
            vision_provider: BedrockProvider with vision support
            
        Returns:
            Dictionary mapping image index to description
        """
        descriptions = {}
        
        logger.info(f"Analyzing {len(images)} images with vision model...")
        
        for i, img in enumerate(images):
            try:
                # Analyze image
                description = vision_provider.generate_with_vision(
                    prompt=(
                        "Describe this image from a document. "
                        "Focus on: charts, diagrams, tables, key information, "
                        "or visual elements. Be concise and factual."
                    ),
                    image_bytes=img.image_bytes,
                    image_media_type=f"image/{img.format}",
                    max_tokens=300
                )
                
                descriptions[i] = description
                logger.info(
                    f"Analyzed image {i+1}/{len(images)} "
                    f"(page {img.page_number})"
                )
                
            except Exception as e:
                logger.warning(f"Failed to analyze image {i}: {e}")
        
        return descriptions
    
    def _analyze_pages_as_images(
        self,
        doc: fitz.Document,
        vision_provider: Any,
        dpi: int = 150
    ) -> Dict[int, str]:
        """
        Render each page as an image and analyze with vision model.
        This captures vector graphics, charts, and diagrams.
        
        Args:
            doc: PyMuPDF document
            vision_provider: BedrockProvider with vision support
            dpi: Resolution for rendering (higher = better quality but slower)
            
        Returns:
            Dictionary mapping page number to description
        """
        page_descriptions = {}
        
        logger.info(
            f"Rendering {len(doc)} pages as images for vision analysis "
            f"(dpi={dpi})..."
        )
        
        for page_num, page in enumerate(doc, 1):
            try:
                # Render page as image
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")
                
                # Analyze with vision
                description = vision_provider.generate_with_vision(
                    prompt=(
                        "Analyze this document page. Describe any charts, graphs, "
                        "diagrams, tables, figures, or important visual elements. "
                        "If it's just text, say 'Text-only page'. Be concise."
                    ),
                    image_bytes=img_bytes,
                    image_media_type="image/png",
                    max_tokens=400
                )
                
                page_descriptions[page_num] = description
                logger.info(f"Analyzed page {page_num}/{len(doc)} as image")
                
            except Exception as e:
                logger.warning(f"Failed to analyze page {page_num} as image: {e}")
        
        return page_descriptions
    
    def _create_chunks(
        self,
        text: str,
        source: str,
        image_descriptions: Optional[Dict[int, str]] = None,
        page_descriptions: Optional[Dict[int, str]] = None
    ) -> List[PDFChunk]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: Full text to chunk
            source: Source PDF filename
            image_descriptions: Optional image descriptions to include
            page_descriptions: Optional page-level descriptions to include
            
        Returns:
            List of PDFChunk objects
        """
        chunks = []
        
        # Add image descriptions to text if available
        visual_context = ""
        if image_descriptions:
            visual_context += "\n\n[EMBEDDED IMAGES IN DOCUMENT]\n"
            for idx, desc in image_descriptions.items():
                visual_context += f"\nImage {idx+1}: {desc}\n"
        
        # Add page descriptions to text if available
        if page_descriptions:
            visual_context += "\n\n[PAGE VISUAL ANALYSIS]\n"
            for page_num, desc in page_descriptions.items():
                # Only include if not just text
                if "text-only" not in desc.lower():
                    visual_context += f"\nPage {page_num}: {desc}\n"
        
        text = text + visual_context
        text_length = len(text)
        
        # Handle empty text
        if text_length == 0:
            return chunks
        
        # If text is smaller than chunk size, return as single chunk
        if text_length <= self.chunk_size:
            chunk = PDFChunk(
                text=text.strip(),
                chunk_id=0,
                page_number=1,
                start_char=0,
                end_char=text_length,
                metadata={
                    'source': source,
                    'chunk_size': text_length,
                    'has_images': bool(image_descriptions),
                    'has_page_analysis': bool(page_descriptions)
                }
            )
            return [chunk]
        
        chunk_id = 0
        start = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Get chunk text
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary (only if not at end)
            if end < text_length and len(chunk_text) > 200:
                # Look for sentence end in last 200 chars
                last_part = chunk_text[-200:]
                last_period = last_part.rfind('. ')
                last_newline = last_part.rfind('\n')
                
                break_point = max(last_period, last_newline)
                
                if break_point != -1:
                    # Adjust end to break at sentence
                    end = start + len(chunk_text) - 200 + break_point + 1
                    chunk_text = text[start:end]
            
            # Create chunk
            chunk = PDFChunk(
                text=chunk_text.strip(),
                chunk_id=chunk_id,
                page_number=self._estimate_page(start, text_length),
                start_char=start,
                end_char=end,
                metadata={
                    'source': source,
                    'chunk_size': len(chunk_text),
                    'has_images': bool(image_descriptions),
                    'has_page_analysis': bool(page_descriptions)
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start >= end:
                break
            
            # Safety check
            if chunk_id > 100:
                logger.warning(f"Created 100 chunks, stopping (text_length={text_length})")
                break
        
        return chunks
    
    def _estimate_page(self, char_pos: int, total_chars: int) -> int:
        """Estimate page number based on character position."""
        # Rough estimate: assume ~2000 chars per page
        return int((char_pos / total_chars) * 10) + 1
    
    def save_images(
        self,
        images: List[PDFImage],
        output_dir: str
    ) -> List[str]:
        """
        Save extracted images to disk.
        
        Args:
            images: List of PDFImage objects
            output_dir: Directory to save images
            
        Returns:
            List of saved image paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for img in images:
            filename = f"page_{img.page_number}_img_{img.image_index}.{img.format}"
            filepath = output_path / filename
            
            with open(filepath, 'wb') as f:
                f.write(img.image_bytes)
            
            saved_paths.append(str(filepath))
            logger.debug(f"Saved image: {filepath}")
        
        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def chunks_to_documents(
        self,
        chunks: List[PDFChunk]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Convert chunks to format suitable for ChromaDB.
        
        Args:
            chunks: List of PDFChunk objects
            
        Returns:
            Tuple of (documents, metadatas)
        """
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                'source': chunk.metadata['source'],
                'chunk_id': chunk.chunk_id,
                'page_number': chunk.page_number,
                'chunk_size': chunk.metadata['chunk_size'],
                'has_images': chunk.metadata['has_images']
            }
            for chunk in chunks
        ]
        
        return documents, metadatas
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PDFParser(chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"extract_images={self.extract_images})"
        )


if __name__ == "__main__":
    # Test the parser
    from ..utils import setup_logger
    
    setup_logger("INFO")
    
    parser = PDFParser(
        chunk_size=1000,
        chunk_overlap=100,
        extract_images=True
    )
    
    print(f"\n✅ Parser initialized: {parser}\n")
    print("="*60)
    print("✅ PDF Parser ready for testing!")
    print("="*60)