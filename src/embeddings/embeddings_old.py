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

    def similarity_search_with_long_query(self, query: str, top_k: int = 5, 
                                    min_similarity: float = 0.0,
                                    aggregation_method: str = "max") -> List[Dict[str, Any]]:
        """
        Find similar documents to a query (handles both short and long queries)
        
        Args:
            query: User query string (short phrase or long patent description)
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            aggregation_method: How to combine similarities from multiple query chunks 
                            ("max" or "average")
            
        Returns:
            List of most similar documents with similarity scores
        """
        if not self.processed_documents:
            logger.warning("No processed documents available for search")
            return []
        
        # Embed the query (may return single embedding or list of embeddings)
        query_embeddings = self.embed_query(query, chunk_if_needed=True)
        
        # Handle single embedding (short query)
        if isinstance(query_embeddings, np.ndarray):
            return self._search_with_single_embedding(query_embeddings, top_k, min_similarity)
        
        # Handle multiple embeddings (long query with chunks)
        else:
            return self._search_with_multiple_embeddings(
                query_embeddings, top_k, min_similarity, aggregation_method
            )

    def _search_with_single_embedding(self, query_embedding: np.ndarray, 
                                    top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Search with a single query embedding"""
        similarities = []
        
        for doc in self.processed_documents:
            doc_embedding = doc["embedding"]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            if similarity >= min_similarity:
                similarities.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": float(similarity)
                })
        
        # Sort and return top results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(similarities)} results above threshold {min_similarity}")
        return similarities[:top_k]

    def _search_with_multiple_embeddings(self, query_embeddings: List[np.ndarray], 
                                    top_k: int, min_similarity: float,
                                    aggregation_method: str) -> List[Dict[str, Any]]:
        """Search with multiple query embeddings (from chunked long query)"""
        # Calculate similarities for each query chunk against all documents
        doc_similarities = {}  # doc_index -> list of similarities
        
        for query_idx, query_embedding in enumerate(query_embeddings):
            logger.debug(f"Processing query chunk {query_idx + 1}/{len(query_embeddings)}")
            
            for doc_idx, doc in enumerate(self.processed_documents):
                doc_embedding = doc["embedding"]
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if doc_idx not in doc_similarities:
                    doc_similarities[doc_idx] = []
                doc_similarities[doc_idx].append(similarity)
        
        # Aggregate similarities for each document
        final_similarities = []
        for doc_idx, similarities_list in doc_similarities.items():
            if aggregation_method == "max":
                final_similarity = max(similarities_list)
            elif aggregation_method == "average":
                final_similarity = sum(similarities_list) / len(similarities_list)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            if final_similarity >= min_similarity:
                doc = self.processed_documents[doc_idx]
                final_similarities.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": float(final_similarity),
                    "query_chunk_similarities": similarities_list  # For debugging
                })
        
        # Sort and return top results
        final_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(final_similarities)} results above threshold {min_similarity}")
        logger.info(f"Used {aggregation_method} aggregation across {len(query_embeddings)} query chunks")            
        return final_similarities[:top_k]
    
    def embed_query(self, query: str, chunk_if_needed: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Embed a query string, with optional chunking for long queries
        
        Args:
            query: User query string (could be a full patent description)
            chunk_if_needed: Whether to chunk long queries
            
        Returns:
            Single embedding vector for short queries, or list of embeddings for chunked long queries
        """
        try:
            query_tokens = self.count_tokens(query)
            logger.debug(f"Query token count: {query_tokens}")
            
            # If query is short, embed directly
            if query_tokens <= self.config.base_chunk_size or not chunk_if_needed:
                embedding = self.embedding_model.encode([query])[0]
                
                # Handle potential NaN values
                if np.isnan(embedding).any():
                    logger.warning("⚠️ NaN values detected in query embedding")
                    embedding = np.nan_to_num(embedding)
                    
                logger.debug(f"Query embedded directly. Shape: {embedding.shape}")
                return embedding
            
            # For long queries, chunk them like documents
            else:
                logger.info(f"Query is long ({query_tokens} tokens), chunking before embedding")
                chunk_size = self.get_chunk_size(query, query_tokens)
                chunks = self.chunk_text_by_tokens(query, chunk_size)
                
                # Embed all chunks
                embeddings = []
                for batch_idx in range(0, len(chunks), self.config.batch_size):
                    batch = chunks[batch_idx:batch_idx + self.config.batch_size]
                    batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                    embeddings.extend(batch_embeddings)
                
                # Handle NaN values
                for i, embedding in enumerate(embeddings):
                    if np.isnan(embedding).any():
                        logger.warning(f"⚠️ NaN values detected in query chunk {i}")
                        embeddings[i] = np.nan_to_num(embedding)
                
                logger.debug(f"Query chunked into {len(embeddings)} chunks")
                return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise EmbeddingError(f"Failed to embed query: {str(e)}") from e

    def __repr__(self) -> str:
        """String representation of the DocumentEmbedder"""
        return (f"DocumentEmbedder(model={self.config.model_name}, "
                f"documents={len(self.documents)}, "
                f"processed={len(self.processed_documents)})")




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
