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
import numpy as np
from transformers import AutoTokenizer
import json
from langchain.docstore.document import Document

from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class DocumentEmbedder:
    """Document embedding and chunking functionality for patent documents"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the document embedder with specified model"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.file_paths = []
        self.documents = []
        self.processed_documents = []
        
    def add_file_paths(self, paths: List[str]) -> None:
        """
        Add file paths to be processed
        
        Args:
            paths: List of file paths to process
        """
        self.file_paths.extend(paths)
        print(f"Added {len(paths)} files. Total files to process: {len(self.file_paths)}")
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add LangChain Document objects to be processed
        
        Args:
            documents: List of Document objects
        """
        self.documents.extend(documents)
        print(f"Added {len(documents)} documents. Total documents to process: {len(self.documents)}")
        
    def load_json_files(self, extract_fn: Callable[[Dict], List[Document]] = None) -> None:
        """
        Load documents from JSON files in self.file_paths
        
        Args:
            extract_fn: Optional function to extract documents from JSON data. 
                        Should accept a dict and return a list of Document objects.
        """
        if not self.file_paths:
            print("No file paths to load. Use add_file_paths() method first.")
            return
            
        loaded_docs = []
        for file_path in self.file_paths:
            try:
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
                    
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if extract_fn is not None:
                    # Use custom extraction function
                    docs = extract_fn(data)
                    loaded_docs.extend(docs)
                else:
                    # Default processing: create a document with the whole JSON content
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
                        
                    doc = Document(page_content=text, metadata=metadata)
                    loaded_docs.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        self.documents.extend(loaded_docs)
        print(f"Loaded {len(loaded_docs)} documents from {len(self.file_paths)} files.")
        print(f"Total documents in memory: {len(self.documents)}")
    
    def process_all_documents(self, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Process all documents in self.documents, generate embeddings and return results
        
        Args:
            debug: Whether to print debug information
            
        Returns:
            List of processed documents with embeddings
        """
        if not self.documents:
            print("No documents to process. Use add_documents() or load_json_files() first.")
            return []
            
        self.processed_documents = self.embed_documents_with_metadata(self.documents, debug=debug)
        print(f"✅ Processed {len(self.processed_documents)} document chunks from {len(self.documents)} documents")
        return self.processed_documents
    
    def save_processed_documents(self, output_path: str) -> None:
        """
        Save processed documents to a file
        
        Args:
            output_path: Path to save the processed documents
        """
        if not self.processed_documents:
            print("No processed documents to save. Run process_all_documents() first.")
            return
            
        # Convert numpy arrays to lists for JSON serialization
        serializable_docs = []
        for doc in self.processed_documents:
            doc_copy = doc.copy()
            doc_copy["embedding"] = doc["embedding"].tolist()
            serializable_docs.append(doc_copy)
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f)
            
        print(f"✅ Saved {len(self.processed_documents)} processed documents to {output_path}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def get_chunk_size(self, text: str, total_tokens: int, base_chunk_size: int = 350, min_chunk_size: int = 180) -> int:
        """Calculate optimal chunk size for text based on token count"""
        if total_tokens <= base_chunk_size:
            return total_tokens

        num_splits = total_tokens // base_chunk_size
        if total_tokens % base_chunk_size != 0:
            num_splits += 1
        
        balanced_chunk_size = math.ceil(total_tokens / num_splits)
        balanced_chunk_size = int(max(min_chunk_size, balanced_chunk_size))
        return balanced_chunk_size

    def chunk_text_by_tokens(self, text: str, chunk_size: int, overlap: int = 30) -> List[str]:
        """Split text into chunks by token count with overlap"""
        print("Starting tokenization...")
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        print(f"Tokenized text length: {len(input_ids)} tokens")
        
        # Safety check to prevent infinite loops
        if chunk_size <= overlap:
            overlap = 0
        
        chunks = []
        start = 0
        
        while start < len(input_ids):
            print(f"Processing chunk at position {start}/{len(input_ids)}")
            end = min(start + chunk_size, len(input_ids))
            chunk_ids = input_ids[start:end]
            
            try:
                chunk_text = self.tokenizer.decode(chunk_ids)
                chunks.append(chunk_text)
            except Exception as e:
                print(f"Error decoding chunk: {e}")
                # Skip this chunk on error
            
            # Advance position
            start += chunk_size - overlap
            
            # Safety check to prevent memory issues
            if len(chunks) > 1000:
                print("⚠️ Warning: Reached maximum number of chunks")
                break
        
        print(f"Created {len(chunks)} chunks")
        return chunks

    def embed_document(self, text: str) -> List[tuple]:
        """Embed a single document, returning list of (chunk_text, embedding_vector) tuples"""
        total_tokens = self.count_tokens(text)
        chunk_size = self.get_chunk_size(text, total_tokens)

        chunks = self.chunk_text_by_tokens(text, chunk_size)
        embeddings = self.embedding_model.encode(chunks)

        return list(zip(chunks, embeddings))

    def embed_documents_with_metadata(self, documents: List[Document], debug: bool = True) -> List[Dict[str, Any]]:
        """Process a list of Document objects, preserving metadata"""
        if debug:
            print(f"📋 Starting document embedding process for {len(documents)} documents")
            start_time = time.time()
        
        processed_docs = []
        
        # Track statistics for debugging
        total_tokens = 0
        total_chunks = 0
        errors = 0
        
        for doc_idx, doc in enumerate(documents):
            if debug and doc_idx % 10 == 0:
                print(f"⏳ Processing document {doc_idx+1}/{len(documents)} ({(doc_idx+1)/len(documents)*100:.1f}%)")
            
            try:
                # Get text content and metadata
                text = doc.page_content
                metadata = doc.metadata
                
                if debug and doc_idx < 3:
                    print(f"\n📄 Document {doc_idx+1} Preview:")
                    print(f"  - Content: {text[:100]}..." if len(text) > 100 else f"  - Content: {text}")
                    print(f"  - Metadata: {metadata}")

                # Calculate chunk size
                doc_tokens = self.count_tokens(text)
                total_tokens += doc_tokens
                chunk_size = self.get_chunk_size(text, doc_tokens)
                
                if debug and doc_idx < 3:
                    print(f"  - Token count: {doc_tokens}")
                    print(f"  - Calculated chunk size: {chunk_size}")
                
                # Create chunks
                chunk_start_time = time.time()
                if doc_tokens < 350:
                    overlap = 0
                else:
                    overlap = 32
                chunks = self.chunk_text_by_tokens(text, chunk_size, overlap)
                total_chunks += len(chunks)
                
                if debug and doc_idx < 3:
                    chunk_time = time.time() - chunk_start_time
                    print(f"  - Chunks created: {len(chunks)} (in {chunk_time:.2f}s)")
                    if chunks:
                        print(f"  - First chunk: {chunks[0][:50]}...")
                
                # Generate embeddings for all chunks
                embedding_start_time = time.time()
                try:
                    batch_size = 8 
                    all_embeddings = []
                    
                    for batch_idx in range(0, len(chunks), batch_size):
                        batch = chunks[batch_idx:batch_idx+batch_size]
                        print(f"  - Processing batch {batch_idx//batch_size + 1}/{(len(chunks)+batch_size-1)//batch_size}")
                        batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                        all_embeddings.extend(batch_embeddings)
                    
                    embeddings = all_embeddings
                    print("✓ Embedding completed")
                except Exception as e:
                    print(f"❌ Error during embedding: {str(e)}")
                    raise
                
                if debug and doc_idx < 3:
                    embedding_time = time.time() - embedding_start_time
                    print(f"  - Embeddings generated: {len(embeddings)} vectors of shape {embeddings[0].shape}")
                    print(f"  - Embedding time: {embedding_time:.2f}s ({embedding_time/len(chunks):.4f}s per chunk)")
                
                # Create new documents with chunks and metadata
                for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                    # Create a copy of metadata and add chunk information
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["source_doc_idx"] = doc_idx
                    
                    # Check for NaN values in embedding
                    if np.isnan(embedding).any():
                        if debug:
                            print(f"⚠️ Warning: NaN values detected in embedding for document {doc_idx}, chunk {i}")
                        # Replace NaN with zeros
                        embedding = np.nan_to_num(embedding)
                    
                    # Create processed document entry
                    processed_docs.append({
                        "text": chunk_text,
                        "embedding": embedding,
                        "metadata": chunk_metadata
                    })
                    
            except Exception as e:
                errors += 1
                if debug:
                    print(f"❌ Error processing document {doc_idx}: {str(e)}")
        
        if debug:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n✅ Embedding completed in {total_time:.2f}s")
            print(f"📊 Statistics:")
            print(f"  - Documents processed: {len(documents)}")
            print(f"  - Total tokens: {total_tokens}")
            print(f"  - Total chunks created: {total_chunks}")
            print(f"  - Average chunks per document: {total_chunks/len(documents):.2f}")
            print(f"  - Processing speed: {total_tokens/total_time:.2f} tokens/second")
            print(f"  - Documents with errors: {errors}")
            print(f"  - Success rate: {(len(documents)-errors)/len(documents)*100:.2f}%")
            
            # Memory usage of embeddings
            embedding_size = sum(emb["embedding"].nbytes for emb in processed_docs)
            print(f"  - Embedding memory usage: {embedding_size/1024/1024:.2f} MB")
        
        return processed_docs

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
def batch_process_json_files(file_paths: List[str], extract_documents_func, batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Process multiple JSON files in batches and generate embeddings.
    
    Args:
        file_paths: List of paths to JSON files
        extract_documents_func: Function to extract documents from JSON data
        batch_size: Number of files to process in each batch
    
    Returns:
        List of processed documents with embeddings
    """
    embedder = DocumentEmbedder()
    all_documents = []
    
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(file_paths)+batch_size-1)//batch_size}")
        
        batch_documents = []
        for file_path in batch_files:
            try:
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract documents (returns list of Document objects)
                    docs = extract_documents_func(data)
                    batch_documents.extend(docs)
                    
                print(f"Loaded {len(docs)} documents from {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Process batch of documents
        embedder.add_documents(batch_documents)
        processed_docs = embedder.process_all_documents(debug=False)
        all_documents.extend(processed_docs)
        
        # Clear documents to free memory
        embedder.documents = []
    
    return all_documents