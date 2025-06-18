"""
Embedding module for processing documents and generating embeddings.
Handles tokenization, chunking, and embedding generation for patent documents.
"""

import time
import math
import os
import json
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
import logging
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BASE_CHUNK_SIZE = 350
DEFAULT_MIN_CHUNK_SIZE = 180
DEFAULT_CHUNK_OVERLAP = 30
SMALL_DOC_THRESHOLD = 350
LARGE_DOC_OVERLAP = 32
DEFAULT_BATCH_SIZE = 8
MAX_CHUNKS_LIMIT = 1000
DEBUG_PREVIEW_LIMIT = 3
PROGRESS_UPDATE_INTERVAL = 10
MEMORY_UNIT_MB = 1024 * 1024

@dataclass
class EmbeddingConfig:
    """Configuration for document embedding process"""
    model_name: str = DEFAULT_MODEL_NAME
    base_chunk_size: int = DEFAULT_BASE_CHUNK_SIZE
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    batch_size: int = DEFAULT_BATCH_SIZE
    max_chunks: int = MAX_CHUNKS_LIMIT
    enable_debug: bool = True
    progress_interval: int = PROGRESS_UPDATE_INTERVAL
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.base_chunk_size <= 0:
            raise ValueError("base_chunk_size must be positive")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

class EmbeddingError(Exception):
    """Exception raised for errors in the embedding process"""
    pass

class DocumentEmbedder:
    """Document embedding and chunking functionality for patent documents"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the document embedder with configuration"""
        self.config = config or EmbeddingConfig()
        self._initialize_models()
        self.file_paths: List[str] = []
        self.documents: List[Document] = []
        self.processed_documents: List[Dict[str, Any]] = []
        
    def _initialize_models(self) -> None:
        """Initialize tokenizer and embedding model"""
        try:
            logger.info(f"Initializing models with: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.embedding_model = SentenceTransformer(self.config.model_name)
            logger.info("Models initialized successfully")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize models: {str(e)}") from e
        
    def add_file_paths(self, paths: List[str]) -> None:
        """
        Add file paths to be processed
        
        Args:
            paths: List of file paths to process
        """
        if not paths:
            logger.warning("Empty path list provided")
            return
            
        valid_paths = [p for p in paths if os.path.exists(p)]
        invalid_paths = [p for p in paths if not os.path.exists(p)]
        
        if invalid_paths:
            logger.warning(f"Found {len(invalid_paths)} invalid paths: {invalid_paths[:5]}...")
            
        self.file_paths.extend(valid_paths)
        logger.info(f"Added {len(valid_paths)} valid files. Total files to process: {len(self.file_paths)}")
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add LangChain Document objects to be processed
        
        Args:
            documents: List of Document objects
        """
        if not documents:
            logger.warning("Empty document list provided")
            return
            
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total documents to process: {len(self.documents)}")
        
    def load_json_files(self, extract_fn: Optional[Callable[[Dict], List[Document]]] = None) -> None:
        """
        Load documents from JSON files in self.file_paths
        
        Args:
            extract_fn: Optional function to extract documents from JSON data
        """
        if not self.file_paths:
            logger.warning("No file paths to load. Use add_file_paths() method first.")
            return
            
        loaded_docs = []
        errors = 0
        
        for file_path in self.file_paths:
            try:
                loaded_docs.extend(self._load_single_json_file(file_path, extract_fn))
            except Exception as e:
                errors += 1
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.documents.extend(loaded_docs)
        logger.info(f"Loaded {len(loaded_docs)} documents from {len(self.file_paths)} files "
                   f"({errors} errors)")
        logger.info(f"Total documents in memory: {len(self.documents)}")
    
    def _load_single_json_file(self, file_path: str, extract_fn: Optional[Callable] = None) -> List[Document]:
        """Load documents from a single JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if extract_fn is not None:
            return extract_fn(data)
        else:
            return self._default_json_extraction(data, file_path)
    
    def _default_json_extraction(self, data: Dict, file_path: str) -> List[Document]:
        """Default JSON extraction logic"""
        metadata = {"source": file_path, "file_type": "json"}
        
        if isinstance(data, dict) and "content" in data:
            text = data["content"]
            # Add other fields as metadata
            for key, value in data.items():
                if key != "content" and not isinstance(value, (dict, list)):
                    metadata[key] = value
        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
            # Add other fields as metadata
            for key, value in data.items():
                if key != "text" and not isinstance(value, (dict, list)):
                    metadata[key] = value
        else:
            # Convert the whole JSON to string as fallback
            text = json.dumps(data)
            
        return [Document(page_content=text, metadata=metadata)]
    
    def process_all_documents(self, debug: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Process all documents in self.documents, generate embeddings and return results
        
        Args:
            debug: Whether to print debug information (overrides config setting)
            
        Returns:
            List of processed documents with embeddings
        """
        if not self.documents:
            logger.warning("No documents to process. Use add_documents() or load_json_files() first.")
            return []
        
        debug_mode = debug if debug is not None else self.config.enable_debug    
        self.processed_documents = self.embed_documents_with_metadata(self.documents, debug=debug_mode)
        logger.info(f"✅ Processed {len(self.processed_documents)} document chunks from {len(self.documents)} documents")
        return self.processed_documents
    
    def save_processed_documents(self, output_path: str) -> None:
        """
        Save processed documents to a file
        
        Args:
            output_path: Path to save the processed documents
        """
        if not self.processed_documents:
            logger.warning("No processed documents to save. Run process_all_documents() first.")
            return
            
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_docs = self._convert_embeddings_to_serializable(self.processed_documents)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, indent=2)
                
            logger.info(f"✅ Saved {len(self.processed_documents)} processed documents to {output_path}")
        except Exception as e:
            raise EmbeddingError(f"Failed to save processed documents: {str(e)}") from e
    
    def _convert_embeddings_to_serializable(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert numpy embeddings to lists for JSON serialization"""
        serializable_docs = []
        for doc in docs:
            doc_copy = doc.copy()
            doc_copy["embedding"] = doc["embedding"].tolist()
            serializable_docs.append(doc_copy)
        return serializable_docs

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer"""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback: estimate based on word count
            return len(text.split()) * 1.3  # Rough estimation
    
    def get_chunk_size(self, text: str, total_tokens: int) -> int:
        """Calculate optimal chunk size for text based on token count"""
        if total_tokens <= self.config.base_chunk_size:
            return total_tokens

        num_splits = math.ceil(total_tokens / self.config.base_chunk_size)
        balanced_chunk_size = math.ceil(total_tokens / num_splits)
        
        return max(self.config.min_chunk_size, balanced_chunk_size)

    def chunk_text_by_tokens(self, text: str, chunk_size: int, overlap: Optional[int] = None) -> List[str]:
        """Split text into chunks by token count with overlap"""
        if overlap is None:
            overlap = self.config.chunk_overlap
            
        logger.debug("Starting tokenization...")
        
        try:
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise EmbeddingError(f"Failed to tokenize text: {str(e)}") from e
            
        logger.debug(f"Tokenized text length: {len(input_ids)} tokens")
        
        # Safety check to prevent infinite loops
        if chunk_size <= overlap:
            overlap = 0
            logger.warning("Chunk size too small, setting overlap to 0")
        
        chunks = []
        start = 0
        
        while start < len(input_ids) and len(chunks) < self.config.max_chunks:
            end = min(start + chunk_size, len(input_ids))
            chunk_ids = input_ids[start:end]
            
            try:
                chunk_text = self.tokenizer.decode(chunk_ids)
                chunks.append(chunk_text)
            except Exception as e:
                logger.warning(f"Error decoding chunk at position {start}: {e}")
                # Skip this chunk on error
            
            # Advance position
            start += chunk_size - overlap
        
        if len(chunks) >= self.config.max_chunks:
            logger.warning(f"⚠️ Reached maximum number of chunks ({self.config.max_chunks})")
        
        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def embed_document(self, text: str) -> List[Tuple[str, np.ndarray]]:
        """Embed a single document, returning list of (chunk_text, embedding_vector) tuples"""
        total_tokens = self.count_tokens(text)
        chunk_size = self.get_chunk_size(text, total_tokens)

        chunks = self.chunk_text_by_tokens(text, chunk_size)
        embeddings = self.embedding_model.encode(chunks)

        return list(zip(chunks, embeddings))

    def embed_documents_with_metadata(self, documents: List[Document], debug: bool = True) -> List[Dict[str, Any]]:
        """Process a list of Document objects, preserving metadata"""
        if debug:
            logger.info(f"📋 Starting document embedding process for {len(documents)} documents")
            start_time = time.time()
        
        processed_docs = []
        stats = self._initialize_processing_stats()
        
        for doc_idx, doc in enumerate(documents):
            if debug and doc_idx % self.config.progress_interval == 0:
                progress = (doc_idx + 1) / len(documents) * 100
                logger.info(f"⏳ Processing document {doc_idx+1}/{len(documents)} ({progress:.1f}%)")
            
            try:
                doc_processed = self._process_single_document(doc, doc_idx, debug, stats)
                processed_docs.extend(doc_processed)
                    
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"❌ Error processing document {doc_idx}: {str(e)}")
        
        if debug:
            self._log_processing_statistics(documents, stats, time.time() - start_time, processed_docs)
        
        return processed_docs
    
    def _initialize_processing_stats(self) -> Dict[str, int]:
        """Initialize statistics tracking for processing"""
        return {
            'total_tokens': 0,
            'total_chunks': 0,
            'errors': 0
        }
    
    def _process_single_document(self, doc: Document, doc_idx: int, debug: bool, stats: Dict) -> List[Dict[str, Any]]:
        """Process a single document and return processed chunks"""
        text = doc.page_content
        metadata = doc.metadata
        
        if debug and doc_idx < DEBUG_PREVIEW_LIMIT:
            self._log_document_preview(doc_idx, text, metadata)

        # Calculate chunk parameters
        doc_tokens = self.count_tokens(text)
        stats['total_tokens'] += doc_tokens
        chunk_size = self.get_chunk_size(text, doc_tokens)
        
        if debug and doc_idx < DEBUG_PREVIEW_LIMIT:
            logger.debug(f"  - Token count: {doc_tokens}")
            logger.debug(f"  - Calculated chunk size: {chunk_size}")
        
        # Create chunks
        chunk_start_time = time.time()
        overlap = 0 if doc_tokens < SMALL_DOC_THRESHOLD else LARGE_DOC_OVERLAP
        chunks = self.chunk_text_by_tokens(text, chunk_size, overlap)
        stats['total_chunks'] += len(chunks)
        
        if debug and doc_idx < DEBUG_PREVIEW_LIMIT:
            chunk_time = time.time() - chunk_start_time
            logger.debug(f"  - Chunks created: {len(chunks)} (in {chunk_time:.2f}s)")
            if chunks:
                logger.debug(f"  - First chunk preview: {chunks[0][:50]}...")
        
        # Generate embeddings
        embeddings = self._generate_embeddings_for_chunks(chunks, doc_idx, debug)
        
        # Create processed document entries
        return self._create_processed_documents(chunks, embeddings, metadata, doc_idx)
    
    def _log_document_preview(self, doc_idx: int, text: str, metadata: Dict) -> None:
        """Log document preview for debugging"""
        logger.debug(f"\n📄 Document {doc_idx+1} Preview:")
        preview_text = text[:100] + "..." if len(text) > 100 else text
        logger.debug(f"  - Content: {preview_text}")
        logger.debug(f"  - Metadata: {metadata}")
    
    def _generate_embeddings_for_chunks(self, chunks: List[str], doc_idx: int, debug: bool) -> List[np.ndarray]:
        """Generate embeddings for document chunks"""
        embedding_start_time = time.time()
        
        try:
            all_embeddings = []
            
            for batch_idx in range(0, len(chunks), self.config.batch_size):
                batch = chunks[batch_idx:batch_idx + self.config.batch_size]
                batch_num = batch_idx // self.config.batch_size + 1
                total_batches = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size
                
                if debug:
                    logger.debug(f"  - Processing batch {batch_num}/{total_batches}")
                
                batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
            
            if debug:
                logger.debug("✓ Embedding completed")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Error during embedding for document {doc_idx}: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e
    
    def _create_processed_documents(self, chunks: List[str], embeddings: List[np.ndarray], 
                                  metadata: Dict, doc_idx: int) -> List[Dict[str, Any]]:
        """Create processed document entries from chunks and embeddings"""
        processed_docs = []
        
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a copy of metadata and add chunk information
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_doc_idx": doc_idx
            })
            
            # Handle NaN values in embedding
            if np.isnan(embedding).any():
                logger.warning(f"⚠️ NaN values detected in embedding for document {doc_idx}, chunk {i}")
                embedding = np.nan_to_num(embedding)
            
            processed_docs.append({
                "text": chunk_text,
                "embedding": embedding,
                "metadata": chunk_metadata
            })
                    
        return processed_docs
    
    def _log_processing_statistics(self, documents: List[Document], stats: Dict, 
                                 total_time: float, processed_docs: List[Dict]) -> None:
        """Log comprehensive processing statistics"""
        logger.info(f"\n✅ Embedding completed in {total_time:.2f}s")
        logger.info("📊 Statistics:")
        logger.info(f"  - Documents processed: {len(documents)}")
        logger.info(f"  - Total tokens: {stats['total_tokens']}")
        logger.info(f"  - Total chunks created: {stats['total_chunks']}")
        
        if len(documents) > 0:
            logger.info(f"  - Average chunks per document: {stats['total_chunks']/len(documents):.2f}")
            logger.info(f"  - Processing speed: {stats['total_tokens']/total_time:.2f} tokens/second")
            success_rate = (len(documents) - stats['errors']) / len(documents) * 100
            logger.info(f"  - Success rate: {success_rate:.2f}%")
        
        logger.info(f"  - Documents with errors: {stats['errors']}")
        
        # Memory usage of embeddings
        if processed_docs:
            embedding_size = sum(doc["embedding"].nbytes for doc in processed_docs)
            logger.info(f"  - Embedding memory usage: {embedding_size/MEMORY_UNIT_MB:.2f} MB")
    
    def __repr__(self) -> str:
        """String representation of the DocumentEmbedder"""
        return (f"DocumentEmbedder(model={self.config.model_name}, "
                f"documents={len(self.documents)}, "
                f"processed={len(self.processed_documents)})")

# Prepare .json file for embedding
def extract_documents_epo(json_data):
    bibliographic = json_data.get("bibliographic_data", {})
    doc_id = bibliographic.get("doc_id", "UNKNOWN")
    documents = []

    # Common metadata to propagate
    common_meta = {
        "doc_id": doc_id,
        "language": bibliographic.get("language"),
        "country": bibliographic.get("country"),
        "doc_number": bibliographic.get("doc_number"),
        "application_number": bibliographic.get("application_number"),
        "publication_date": bibliographic.get("publication_date"),
        "ipc_classes": bibliographic.get("ipc_classes", []),
        "file":bibliographic.get("file")
    }

    # Title (en preferred)
    title_dict = bibliographic.get("title", {})
    title = title_dict.get("en") or next(iter(title_dict.values()), "")
    if title:
        documents.append(Document(
            page_content=title,
            metadata={**common_meta, "section": "title"}
        ))

    # Abstract
    abstract = bibliographic.get("abstract")
    if abstract:
        documents.append(Document(
            page_content=abstract,
            metadata={**common_meta, "section": "abstract"}
        ))

    # Claims
    for claim in json_data.get("claims", []):
        documents.append(Document(
            page_content=claim["text"],
            metadata={**common_meta, "section": "claim", "claim_number": claim.get("claim_number")}
        ))

    # Main sections
    for section in json_data.get("main_sections", []):
        section_name = section.get("heading_text", "UNKNOWN_SECTION")
        for p in section.get("paragraphs", []):
            documents.append(Document(
                page_content=f"{section_name}\n{p['text']}",
                metadata={**common_meta, "section": section_name, "p_id": p.get("p_id")}
            ))

    return documents


# Update the batch processing function to use the DocumentEmbedder class
def batch_process_json_files(file_paths: List[str], extract_documents_func: Callable, 
                           batch_size: int = 10, config: Optional[EmbeddingConfig] = None) -> List[Dict[str, Any]]:
    """
    Process multiple JSON files in batches and generate embeddings.
    
    Args:
        file_paths: List of paths to JSON files
        extract_documents_func: Function to extract documents from JSON data
        batch_size: Number of files to process in each batch
        config: Optional embedding configuration
    
    Returns:
        List of processed documents with embeddings
    """
    if not file_paths:
        logger.warning("No file paths provided for batch processing")
        return []
        
    embedder = DocumentEmbedder(config or EmbeddingConfig())
    all_documents = []
    
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        batch_documents = []
        for file_path in batch_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs = extract_documents_func(data)
                    batch_documents.extend(docs)
                    
                logger.debug(f"Loaded {len(docs)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Process batch of documents
        if batch_documents:
            embedder.add_documents(batch_documents)
            processed_docs = embedder.process_all_documents(debug=False)
            all_documents.extend(processed_docs)
            
            # Clear documents to free memory
            embedder.documents = []
        
    logger.info(f"Batch processing completed. Total processed documents: {len(all_documents)}")
    return all_documents