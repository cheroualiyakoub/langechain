"""
Embedding module for processing documents and generating embeddings.
Integrates with TextChunker for document segmentation.
"""

import time
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

from src.embeddings.text_Chunker import TextChunker

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Exception raised for errors during embedding operations"""
    pass

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32
    enable_debug: bool = False
    progress_interval: int = 10
    use_semantic_chunking: bool = True
    min_similarity_threshold: float = 0.1

class DocumentEmbedder:
    """Clean document embedding functionality that integrates with TextChunker"""

    def __init__(self, chunker: TextChunker, config: Optional[EmbeddingConfig] = None):
        """Initialize the document embedder
        
        Args:
            chunker: TextChunker instance for document chunking
            config: Optional embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.chunker = chunker
        self.embedding_model = self._initialize_model()
        self.documents: List[Document] = []
        self.processed_documents: List[Dict[str, Any]] = []
        logger.info(f"DocumentEmbedder initialized with model {self.config.model_name}")
        
    def _initialize_model(self) -> SentenceTransformer:
        """Initialize embedding model"""
        try:
            logger.info(f"Initializing embedding model: {self.config.model_name}")
            model = SentenceTransformer(self.config.model_name)
            logger.info(f"Model initialized successfully with vector size {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embedding model: {str(e)}") from e
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to be processed"""
        if not documents:
            logger.warning("Empty document list provided")
            return
            
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total documents: {len(self.documents)}")
    
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents and generate embeddings
        
        Returns:
            List of processed documents with embeddings
        """
        if not self.documents:
            logger.warning("No documents to process")
            return []
        
        debug_mode = self.config.enable_debug
        start_time = time.time()
        processed_docs = []
        total_chunks = 0
        
        for doc_idx, doc in enumerate(self.documents):
            if debug_mode and doc_idx % self.config.progress_interval == 0:
                progress = (doc_idx + 1) / len(self.documents) * 100
                logger.info(f"Processing document {doc_idx+1}/{len(self.documents)} ({progress:.1f}%)")
            
            try:
                # Chunk the document
                text = doc.page_content
                chunks = self.chunker.chunk_text(text, use_semantic=self.config.use_semantic_chunking)
                
                # Generate embeddings
                embeddings = self._generate_embeddings_for_chunks(chunks)
                
                # Store processed chunks
                doc_chunks = self._create_processed_documents(chunks, embeddings, doc.metadata, doc_idx)
                processed_docs.extend(doc_chunks)
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"Error processing document {doc_idx}: {str(e)}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f}s")
        logger.info(f"Created {total_chunks} chunks from {len(self.documents)} documents")
        
        self.processed_documents = processed_docs
        return processed_docs

    def _generate_embeddings_for_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """Generate embeddings for text chunks"""
        embeddings = []
        
        for batch_idx in range(0, len(chunks), self.config.batch_size):
            batch = chunks[batch_idx:batch_idx + self.config.batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            # Handle potential NaN values
            for i, embedding in enumerate(batch_embeddings):
                if np.isnan(embedding).any():
                    logger.warning(f"NaN values detected in embedding at position {batch_idx + i}")
                    embeddings[batch_idx + i] = np.nan_to_num(embedding)
                
        return embeddings
    
    def _create_processed_documents(self, chunks: List[str], embeddings: List[np.ndarray], 
                                   metadata: Dict, doc_idx: int) -> List[Dict[str, Any]]:
        """Create processed document entries from chunks and embeddings"""
        processed_docs = []
        
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_doc_idx": doc_idx
            })
            
            processed_docs.append({
                "text": chunk_text,
                "embedding": embedding,
                "metadata": chunk_metadata
            })
                    
        return processed_docs
        
    def similarity_search(self, query: str, top_k: int = 5, min_similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find similar documents to a query
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (defaults to config value)
            
        Returns:
            List of similar documents with similarity scores
        """
        if not self.processed_documents:
            logger.warning("No processed documents available for search")
            return []
            
        threshold = min_similarity if min_similarity is not None else self.config.min_similarity_threshold
        
        # Get query embedding
        chunks = self.chunker.chunk_text(query, use_semantic=self.config.use_semantic_chunking)
        query_embeddings = self._generate_embeddings_for_chunks(chunks)
        
        # Single query chunk case
        if len(query_embeddings) == 1:
            return self._search_with_embedding(query_embeddings[0], top_k, threshold)
            
        # Multiple query chunks case
        return self._search_with_multiple_embeddings(query_embeddings, top_k, threshold)
            
    def _search_with_embedding(self, query_embedding: np.ndarray, 
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
        
        # Sort by similarity (descending) and return top results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(similarities)} results above threshold {min_similarity}")
        return similarities[:top_k]

    def _search_with_multiple_embeddings(self, query_embeddings: List[np.ndarray], 
                                        top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Search with multiple query embeddings, using max aggregation"""
        # Calculate max similarity for each document across all query chunks
        best_similarities = {}  # doc_idx -> best similarity score
        
        for query_embedding in query_embeddings:
            for doc_idx, doc in enumerate(self.processed_documents):
                doc_embedding = doc["embedding"]
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                # Keep track of best similarity for each document
                if doc_idx not in best_similarities or similarity > best_similarities[doc_idx]:
                    best_similarities[doc_idx] = similarity
        
        # Filter, format and sort results
        results = []
        for doc_idx, similarity in best_similarities.items():
            if similarity >= min_similarity:
                doc = self.processed_documents[doc_idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": float(similarity)
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} results above threshold {min_similarity}")
        return results[:top_k]

    def save_embeddings(self, output_path: str) -> None:
        """Save embeddings to a file"""
        if not self.processed_documents:
            logger.warning("No processed documents to save")
            return
            
        try:
            serializable_docs = []
            for doc in self.processed_documents:
                doc_copy = doc.copy()
                # Convert numpy arrays to lists for JSON serialization
                doc_copy["embedding"] = doc["embedding"].tolist()
                serializable_docs.append(doc_copy)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, indent=2)
                
            logger.info(f"Saved {len(self.processed_documents)} document embeddings to {output_path}")
        except Exception as e:
            raise EmbeddingError(f"Failed to save embeddings: {str(e)}") from e
            
    def load_embeddings(self, input_path: str) -> None:
        """Load embeddings from a file"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                loaded_docs = json.load(f)
            
            processed_docs = []
            for doc in loaded_docs:
                # Convert lists back to numpy arrays
                doc["embedding"] = np.array(doc["embedding"])
                processed_docs.append(doc)
            
            self.processed_documents = processed_docs
            logger.info(f"Loaded {len(processed_docs)} document embeddings from {input_path}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embeddings: {str(e)}") from e
    
    def clear_documents(self) -> None:
        """Clear all documents to free memory"""
        self.documents = []
        logger.info("Document list cleared")
        
    def __repr__(self) -> str:
        """String representation"""
        return (f"DocumentEmbedder(model={self.config.model_name}, "
                f"documents={len(self.documents)}, "
                f"processed={len(self.processed_documents)})")