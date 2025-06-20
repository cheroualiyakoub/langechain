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
<<<<<<< Updated upstream
                f"processed={len(self.processed_documents)})")
=======
                f"processed={len(self.processed_documents)})")

# Prepare .json file for embedding
def extract_documents_epo(json_data):
    bibliographic = json_data.get("bibliographic_data", {})
    doc_id = bibliographic.get("doc_id", "UNKNOWN")
    documents = []

    # Common metadata to propagate
    common_meta = {
        "doc_id": doc_id,
        "title": bibliographic.get("title", {}).get("en"),
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
>>>>>>> Stashed changes
