"""Improved text chunking functionality for document processing"""

import logging
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer
import math
import re

from config.chunker_config import ChunkerConfig

# Configure logging
logger = logging.getLogger(__name__)

class ChunkerError(Exception):
    """Exception raised for errors during text chunking"""
    pass

class TextChunker:
    """Handles text chunking strategies with semantic awareness"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Optional[ChunkerConfig] = None):
        """Initialize the text chunker
        
        Args:
            tokenizer: Hugging Face tokenizer
            config: ChunkerConfig with chunking parameters
        """
        self.tokenizer = tokenizer
        self.config = config or ChunkerConfig()
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
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
            return int(len(text.split()) * 1.3)  # Rough estimation
    
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
    
    def find_best_split_position(self, text: str, target_pos: int, window: int = 100) -> int:
        """Find the best position to split text near target position
        
        Args:
            text: Text to analyze
            target_pos: Target character position for split
            window: Search window around target position
            
        Returns:
            Best character position to split
        """
        start_search = max(0, target_pos - window)
        end_search = min(len(text), target_pos + window)
        search_text = text[start_search:end_search]
        
        # Look for paragraph breaks first (highest priority)
        paragraph_matches = list(self.paragraph_breaks.finditer(search_text))
        if paragraph_matches:
            # Find closest paragraph break to target
            best_match = min(paragraph_matches, 
                           key=lambda m: abs(m.end() + start_search - target_pos))
            return best_match.end() + start_search
        
        # Look for sentence endings
        sentence_matches = list(self.sentence_endings.finditer(search_text))
        if sentence_matches:
            # Find closest sentence ending to target
            best_match = min(sentence_matches,
                           key=lambda m: abs(m.end() + start_search - target_pos))
            return best_match.end() + start_search
        
        # Fallback: look for word boundaries
        words = search_text.split()
        if len(words) > 1:
            # Find word boundary closest to middle of search window
            mid_char = len(search_text) // 2
            char_pos = 0
            for i, word in enumerate(words[:-1]):  # Don't split after last word
                char_pos += len(word) + 1  # +1 for space
                if char_pos >= mid_char:
                    return start_search + char_pos
        
        # Last resort: return target position
        return target_pos
    
    def chunk_by_semantic_units(self, text: str, chunk_size: int, overlap: Optional[int] = None) -> List[str]:
        """Split text into chunks respecting semantic boundaries
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of text chunks
        """
        if overlap is None:
            overlap = self.config.chunk_overlap
            
        # Quick return for short text
        if self.count_tokens(text) <= chunk_size:
            return [text]
        
        chunks = []
        start_pos = 0
        char_per_token = len(text) / max(self.count_tokens(text), 1)
        
        # Track previous position to detect lack of progress
        prev_start_pos = -1
        
        while start_pos < len(text) and len(chunks) < self.config.max_chunks:
            # Estimate end position based on token density
            estimated_end = start_pos + int(chunk_size * char_per_token)
            estimated_end = min(estimated_end, len(text))
            
            # Find the best semantic boundary for splitting
            if estimated_end < len(text):
                actual_end = self.find_best_split_position(text, estimated_end)
            else:
                actual_end = len(text)
            
            # Extract chunk
            chunk_text = text[start_pos:actual_end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                # Verify token count and adjust if necessary
                chunk_tokens = self.count_tokens(chunk_text)
                
                # If chunk is too large, force split at token boundary
                if chunk_tokens > chunk_size * 1.2:  # 20% tolerance
                    chunk_text = self.force_split_by_tokens(chunk_text, chunk_size)
                
                chunks.append(chunk_text)
            
            # Calculate next start position
            if actual_end >= len(text):
                break
            
            prev_start_pos = start_pos  # Save current position before updating
            
            if overlap == 0:
                # No overlap - start next chunk right after current one
                start_pos = actual_end
            else:
                # Calculate overlap in characters
                overlap_chars = int(overlap * char_per_token)
                start_pos = max(actual_end - overlap_chars, start_pos + 1)
            
            # Safety check to ensure progress
            if start_pos <= prev_start_pos and len(chunks) > 0:  # FIXED: Compare with prev_start_pos
                # If we're not making progress, force advancement
                start_pos = actual_end
        
        if len(chunks) >= self.config.max_chunks:
            logger.warning(f"⚠️ Reached maximum number of chunks ({self.config.max_chunks})")
        
        logger.debug(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def force_split_by_tokens(self, text: str, max_tokens: int) -> str:
        """Force split text to fit within token limit
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text that fits within token limit
        """
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
        except Exception as e:
            logger.warning(f"Error in force split: {e}")
            # Fallback: simple character-based truncation
            estimated_chars = int(max_tokens * len(text) / max(self.count_tokens(text), 1))
            return text[:estimated_chars]
    
    def chunk_by_tokens(self, text: str, chunk_size: int, overlap: Optional[int] = None) -> List[str]:
        """Split text into chunks by token count with improved overlap handling
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            
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
            
        logger.info(f"Tokenized text length: {len(input_ids)} tokens")
        
        if len(input_ids) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        
        while start < len(input_ids) and len(chunks) < self.config.max_chunks:
            end = min(start + chunk_size, len(input_ids))
            chunk_ids = input_ids[start:end]
            
            try:
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                # Clean up subword artifacts
                chunk_text = self._clean_chunk_text(chunk_text)
                
                if chunk_text and chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text.strip())
                    logger.debug(f"Chunk {len(chunks)}: start={start}, end={end}, tokens={len(chunk_ids)}")
            except Exception as e:
                logger.warning(f"Error decoding chunk at position {start}: {e}")
            
            # Calculate advancement
            if overlap == 0:
                # No overlap - advance by full chunk size
                start = end
            else:
                # With overlap - advance by chunk_size minus overlap
                advance = max(chunk_size - overlap, 1)  # Ensure minimum advance of 1
                start += advance
        
        if len(chunks) >= self.config.max_chunks:
            logger.warning(f"⚠️ Reached maximum number of chunks ({self.config.max_chunks})")
        
        logger.debug(f"Created {len(chunks)} token-based chunks")
        return chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean up chunk text to remove tokenization artifacts
        
        Args:
            text: Raw decoded text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
            
        # Remove leading subword markers (##, ▁, etc.)
        text = re.sub(r'^##\w*\s*', '', text)
        text = re.sub(r'^▁+', '', text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # If text starts with a partial word, try to find the first complete word
        words = text.split()
        if words and len(words) > 1:
            # Check if first word looks incomplete (starts with lowercase after cleaning)
            first_word = words[0]
            if first_word and first_word[0].islower() and len(first_word) < 3:
                # Remove potentially incomplete first word
                text = ' '.join(words[1:])
        
        return text.strip()
    
    def chunk_text(self, text: str, use_semantic: bool = True) -> List[str]:
        """Convenience method to chunk text using optimal strategy
        
        Args:
            text: Text to chunk
            use_semantic: Whether to use semantic-aware chunking
            
        Returns:
            List of text chunks
        """
        total_tokens = self.count_tokens(text)
        logger.info(f"Input text has {total_tokens} tokens")
        
        if total_tokens <= self.config.base_chunk_size:
            logger.info("Text fits in single chunk")
            return [text]
            
        chunk_size = self.get_chunk_size(text, total_tokens)
        logger.info(f"Using chunk size: {chunk_size} tokens, overlap: {self.config.chunk_overlap}")
        
        if use_semantic:
            return self.chunk_by_semantic_units(text, chunk_size)
        else:
            return self.chunk_by_tokens(text, chunk_size)
    
    def debug_chunking(self, text: str, use_semantic: bool = True) -> List[str]:
        """Debug version of chunk_text that provides detailed output"""
        total_tokens = self.count_tokens(text)
        print(f"🔍 DEBUG: Input text has {total_tokens} tokens")
        print(f"🔍 DEBUG: Base chunk size: {self.config.base_chunk_size}")
        print(f"🔍 DEBUG: Chunk overlap: {self.config.chunk_overlap}")
        
        if total_tokens <= self.config.base_chunk_size:
            print("🔍 DEBUG: Text fits in single chunk")
            return [text]
            
        chunk_size = self.get_chunk_size(text, total_tokens)
        print(f"🔍 DEBUG: Calculated chunk size: {chunk_size} tokens")
        
        if use_semantic:
            print("🔍 DEBUG: Using semantic chunking")
            chunks = self.chunk_by_semantic_units(text, chunk_size)
        else:
            print("🔍 DEBUG: Using token-based chunking")
            chunks = self.chunk_by_tokens(text, chunk_size)
        
        print(f"🔍 DEBUG: Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunk_tokens = self.count_tokens(chunk)
            print(f"🔍 DEBUG: Chunk {i+1}: {chunk_tokens} tokens, first 50 chars: '{chunk[:50]}...'")
        
        return chunks
    
    def analyze_chunks(self, chunks: List[str]) -> dict:
        """Analyze chunk quality and provide statistics
        
        Args:
            chunks: List of text chunks to analyze
            
        Returns:
            Dictionary with chunk analysis results
        """
        if not chunks:
            return {"error": "No chunks provided"}
        
        token_counts = [self.count_tokens(chunk) for chunk in chunks]
        char_counts = [len(chunk) for chunk in chunks]
        
        # Calculate actual overlaps between adjacent chunks
        overlaps = []
        for i in range(len(chunks) - 1):
            overlap = self._calculate_text_overlap(chunks[i], chunks[i + 1])
            overlaps.append(overlap)
        
        # Check for duplicate chunks
        unique_chunks = set(chunks)
        has_duplicates = len(chunks) != len(unique_chunks)
        
        return {
            "num_chunks": len(chunks),
            "has_duplicates": has_duplicates,
            "unique_chunks": len(unique_chunks),
            "token_stats": {
                "min": min(token_counts),
                "max": max(token_counts),
                "avg": round(sum(token_counts) / len(token_counts), 1),
                "total": sum(token_counts),
                "target_chunk_size": getattr(self.config, 'base_chunk_size', 'unknown')
            },
            "char_stats": {
                "min": min(char_counts),
                "max": max(char_counts),
                "avg": round(sum(char_counts) / len(char_counts), 1)
            },
            "overlap_stats": {
                "avg_overlap_ratio": round(sum(overlaps) / len(overlaps), 3) if overlaps else 0,
                "min_overlap": round(min(overlaps), 3) if overlaps else 0,
                "max_overlap": round(max(overlaps), 3) if overlaps else 0,
                "target_overlap": getattr(self.config, 'chunk_overlap', 'unknown')
            },
            "chunks_preview": [
                {
                    "chunk_num": i+1,
                    "tokens": token_counts[i],
                    "chars": char_counts[i],
                    "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                for i, chunk in enumerate(chunks[:3])
            ]
        }
    
    def _calculate_text_overlap(self, chunk1: str, chunk2: str) -> float:
        """Calculate text overlap between two chunks using word-level comparison"""
        if not chunk1 or not chunk2:
            return 0.0
        
        words1 = chunk1.lower().split()
        words2 = chunk2.lower().split()
        
        if not words1 or not words2:
            return 0.0
        
        # Find the longest common subsequence at the boundaries
        # Check suffix of chunk1 vs prefix of chunk2
        max_possible_overlap = min(len(words1), len(words2))
        overlap_words = 0
        
        for i in range(1, max_possible_overlap + 1):
            if words1[-i:] == words2[:i]:
                overlap_words = i
        
        # Calculate overlap ratio relative to the smaller chunk
        smaller_chunk_size = min(len(words1), len(words2))
        return overlap_words / smaller_chunk_size if smaller_chunk_size > 0 else 0.0