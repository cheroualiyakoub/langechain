

"""Text chunking functionality for document processing"""

import logging
from typing import List, Optional
from transformers import PreTrainedTokenizer
import math

from config.chunker_config import ChunkerConfig

# Configure logging
logger = logging.getLogger(__name__)

class ChunkerError(Exception):
    """Exception raised for errors during text chunking"""
    pass

class TextChunker:
    """Handles text chunking strategies"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Optional[ChunkerConfig] = None):
        """Initialize the text chunker
        
        Args:
            tokenizer: Hugging Face tokenizer
            config: ChunkerConfig with chunking parameters
        """
        self.tokenizer = tokenizer
        self.config = config or ChunkerConfig()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer
        
        Args:
            text: Text to tokenize
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback: estimate based on word count
            return len(text.split()) * 1.3  # Rough estimation
    
    def get_chunk_size(self, text: str, total_tokens: int) -> int:
        """Calculate optimal chunk size for text based on token count
        
        Args:
            text: Text to chunk
            total_tokens: Total number of tokens in text
            
        Returns:
            Optimal chunk size in tokens
        """
        if total_tokens <= self.config.base_chunk_size:
            return total_tokens

        num_splits = math.ceil(total_tokens / self.config.base_chunk_size)
        balanced_chunk_size = math.ceil(total_tokens / num_splits)
        
        return max(self.config.min_chunk_size, balanced_chunk_size)

    def chunk_by_tokens(self, text: str, chunk_size: int, overlap: Optional[int] = None) -> List[str]:
        """Split text into chunks by token count with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens (defaults to config.chunk_overlap)
            
        Returns:
            List of text chunks
        """
        if overlap is None:
            overlap = self.config.chunk_overlap
            
        logger.debug("Starting tokenization...")
        
        try:
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise ChunkerError(f"Failed to tokenize text: {str(e)}") from e
            
        print((f"Tokenized text length: {len(input_ids)} tokens"))
        
        if len(input_ids) <= chunk_size:
         return [text]  

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
    
    def chunk_text(self, text: str) -> List[str]:
        """Convenience method to chunk text using default config settings
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.config.base_chunk_size:
            return [text]
        chunk_size = self.get_chunk_size(text, total_tokens)
        return self.chunk_by_tokens(text, chunk_size)