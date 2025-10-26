"""
Ingest commands for PDF document processing.

Handles:
- Single PDF ingestion with vision analysis
- Batch PDF ingestion with parallel processing
- Progress tracking and error handling
"""

import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core import PDFParser, BedrockProvider, ChromaDBManager, VectorStore
from ..utils import get_logger, ensure_dir

logger = get_logger(__name__)


class IngestCommand:
    """Command handler for PDF ingestion."""
    
    def __init__(self, config):
        """
        Initialize ingest command.
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info("IngestCommand initialized")
    
    def ingest_single(
        self,
        pdf_path: str,
        skip_vision: bool = False,
        collection: Optional[str] = None
    ) -> int:
        """
        Ingest a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            skip_vision: Skip vision analysis
            collection: Override collection name
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        pdf_path = Path(pdf_path)
        
        # Validate file exists
        if not pdf_path.exists():
            print(f"❌ Error: PDF not found: {pdf_path}")
            return 1
        
        if not pdf_path.suffix.lower() == '.pdf':
            print(f"❌ Error: File is not a PDF: {pdf_path}")
            return 1
        
        print(f"\n{'='*80}")
        print(f"INGESTING PDF DOCUMENT")
        print(f"{'='*80}")
        print(f"\n📄 File: {pdf_path.name}")
        print(f"   Path: {pdf_path.absolute()}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        # Prompt for confirmation
        response = input(f"\nProceed with ingestion? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Ingestion cancelled")
            return 1
        
        try:
            # Initialize components
            print(f"\n{'─'*80}")
            print("STEP 1: Initializing Components")
            print(f"{'─'*80}")
            
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                embedding_model_id=self.config.bedrock.embedding_model_id,
                region=self.config.bedrock.region
            )
            print("✅ Bedrock provider initialized")
            
            parser = PDFParser(
                chunk_size=self.config.pdf_processing.chunk_size,
                chunk_overlap=self.config.pdf_processing.chunk_overlap,
                extract_images=self.config.pdf_processing.extract_images
            )
            print("✅ PDF parser initialized")
            
            collection_name = collection or self.config.chromadb.collection_name
            db_manager = ChromaDBManager(
                persist_directory=self.config.chromadb.persist_directory,
                collection_name=collection_name
            )
            print(f"✅ ChromaDB manager initialized (collection: {collection_name})")
            
            # Parse PDF
            print(f"\n{'─'*80}")
            print("STEP 2: Parsing PDF")
            print(f"{'─'*80}")
            
            analyze_images = not skip_vision and self.config.pdf_processing.use_vision_for_images
            
            if analyze_images:
                print("⚙️  Vision analysis: ENABLED")
                print("   - Embedded images: Will be analyzed")
                print("   - Page-level (charts/graphs): Will be analyzed")
                print("   (This may take several minutes for large PDFs)")
            else:
                print("⚙️  Vision analysis: DISABLED")
            
            print("\nParsing document...")
            
            result = parser.parse_pdf(
                str(pdf_path),
                vision_provider=provider if analyze_images else None,
                analyze_images=analyze_images,
                analyze_pages=analyze_images  # ← CRITICAL: Enable page-level vision!
            )
            
            print(f"\n✅ PDF parsed successfully!")
            print(f"   Pages: {result['total_pages']}")
            print(f"   Text length: {len(result['full_text']):,} characters")
            print(f"   Images extracted: {len(result['images'])}")
            print(f"   Text chunks: {len(result['chunks'])}")
            
            if analyze_images and result['image_descriptions']:
                print(f"   Images analyzed: {len(result['image_descriptions'])}")
            
            # Generate embeddings
            print(f"\n{'─'*80}")
            print("STEP 3: Generating Embeddings")
            print(f"{'─'*80}")
            
            documents, metadatas = parser.chunks_to_documents(result['chunks'])
            
            print(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = provider.generate_embeddings_batch(documents)
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            
            # Store in ChromaDB
            print(f"\n{'─'*80}")
            print("STEP 4: Storing in ChromaDB")
            print(f"{'─'*80}")
            
            print("Adding documents to vector database...")
            
            ids = db_manager.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"\n✅ Stored {len(ids)} chunks in ChromaDB")
            print(f"   Collection: {collection_name}")
            print(f"   Total documents: {db_manager.count()}")
            
            # Summary
            print(f"\n{'='*80}")
            print("INGESTION COMPLETE")
            print(f"{'='*80}")
            print(f"\n✅ Successfully ingested: {pdf_path.name}")
            print(f"\n📊 Summary:")
            print(f"   Chunks created: {len(result['chunks'])}")
            print(f"   Chunks stored: {len(ids)}")
            print(f"   Collection: {collection_name}")
            print(f"   Total in DB: {db_manager.count()} documents")
            
            logger.info(f"Successfully ingested {pdf_path.name}")
            
            return 0
        
        except Exception as e:
            print(f"\n❌ Error during ingestion: {e}")
            logger.error(f"Ingestion failed for {pdf_path}: {e}", exc_info=True)
            return 1
    
    def ingest_batch(
        self,
        directory: str,
        recursive: bool = False,
        parallel: int = 1,
        skip_vision: bool = False,
        collection: Optional[str] = None
    ) -> int:
        """
        Ingest all PDFs in a directory.
        
        Args:
            directory: Directory containing PDFs
            recursive: Search recursively
            parallel: Number of parallel workers
            skip_vision: Skip vision analysis
            collection: Override collection name
            
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        directory = Path(directory)
        
        # Validate directory exists
        if not directory.exists():
            print(f"❌ Error: Directory not found: {directory}")
            return 1
        
        if not directory.is_dir():
            print(f"❌ Error: Not a directory: {directory}")
            return 1
        
        # Find PDFs
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ Error: No PDF files found in {directory}")
            return 1
        
        print(f"\n{'='*80}")
        print(f"BATCH PDF INGESTION")
        print(f"{'='*80}")
        print(f"\n📂 Directory: {directory.absolute()}")
        print(f"   PDFs found: {len(pdf_files)}")
        print(f"   Recursive: {'Yes' if recursive else 'No'}")
        print(f"   Parallel workers: {parallel}")
        print(f"   Vision analysis: {'Disabled' if skip_vision else 'Enabled'}")
        
        # Show files
        print(f"\n📄 Files to ingest:")
        for i, pdf in enumerate(pdf_files[:10], 1):
            print(f"   {i}. {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
        
        if len(pdf_files) > 10:
            print(f"   ... and {len(pdf_files) - 10} more")
        
        # Prompt for confirmation
        response = input(f"\nProceed with batch ingestion of {len(pdf_files)} PDFs? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Batch ingestion cancelled")
            return 1
        
        # Initialize components (shared across workers)
        try:
            provider = BedrockProvider(
                model_id=self.config.bedrock.model_id,
                embedding_model_id=self.config.bedrock.embedding_model_id,
                region=self.config.bedrock.region
            )
            
            collection_name = collection or self.config.chromadb.collection_name
            db_manager = ChromaDBManager(
                persist_directory=self.config.chromadb.persist_directory,
                collection_name=collection_name
            )
            
            parser = PDFParser(
                chunk_size=self.config.pdf_processing.chunk_size,
                chunk_overlap=self.config.pdf_processing.chunk_overlap,
                extract_images=self.config.pdf_processing.extract_images
            )
            
            print(f"\n✅ Components initialized")
            print(f"   Collection: {collection_name}")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return 1
        
        # Process PDFs
        print(f"\n{'─'*80}")
        print("PROCESSING PDFs")
        print(f"{'─'*80}\n")
        
        analyze_images = not skip_vision and self.config.pdf_processing.use_vision_for_images
        
        success_count = 0
        failure_count = 0
        failed_files = []
        
        if parallel > 1:
            # Parallel processing
            print(f"⚙️  Using {parallel} parallel workers\n")
            
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        self._process_single_pdf,
                        pdf_file,
                        parser,
                        provider,
                        db_manager,
                        analyze_images
                    ): pdf_file
                    for pdf_file in pdf_files
                }
                
                for future in as_completed(futures):
                    pdf_file = futures[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                            print(f"✅ [{success_count + failure_count}/{len(pdf_files)}] {pdf_file.name}")
                        else:
                            failure_count += 1
                            failed_files.append(pdf_file.name)
                            print(f"❌ [{success_count + failure_count}/{len(pdf_files)}] {pdf_file.name} - FAILED")
                    except Exception as e:
                        failure_count += 1
                        failed_files.append(pdf_file.name)
                        print(f"❌ [{success_count + failure_count}/{len(pdf_files)}] {pdf_file.name} - ERROR: {e}")
        
        else:
            # Sequential processing
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
                
                try:
                    success = self._process_single_pdf(
                        pdf_file,
                        parser,
                        provider,
                        db_manager,
                        analyze_images
                    )
                    
                    if success:
                        success_count += 1
                        print(f"   ✅ Success\n")
                    else:
                        failure_count += 1
                        failed_files.append(pdf_file.name)
                        print(f"   ❌ Failed\n")
                
                except Exception as e:
                    failure_count += 1
                    failed_files.append(pdf_file.name)
                    print(f"   ❌ Error: {e}\n")
        
        # Summary
        print(f"\n{'='*80}")
        print("BATCH INGESTION COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Summary:")
        print(f"   Total PDFs: {len(pdf_files)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {failure_count}")
        print(f"   Collection: {collection_name}")
        print(f"   Total documents in DB: {db_manager.count()}")
        
        if failed_files:
            print(f"\n⚠️  Failed files:")
            for filename in failed_files:
                print(f"   - {filename}")
        
        return 0 if failure_count == 0 else 1
    
    def _process_single_pdf(
        self,
        pdf_file: Path,
        parser: PDFParser,
        provider: BedrockProvider,
        db_manager: ChromaDBManager,
        analyze_images: bool
    ) -> bool:
        """
        Process a single PDF file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse PDF
            result = parser.parse_pdf(
                str(pdf_file),
                vision_provider=provider if analyze_images else None,
                analyze_images=analyze_images,
                analyze_pages=analyze_images  # ← CRITICAL: Enable page-level vision!
            )
            
            # Convert to documents
            documents, metadatas = parser.chunks_to_documents(result['chunks'])
            
            # Generate embeddings
            embeddings = provider.generate_embeddings_batch(documents)
            
            # Store in ChromaDB
            db_manager.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully processed {pdf_file.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            return False