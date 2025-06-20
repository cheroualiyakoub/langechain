"""Configuration for text chunking operations"""

from dataclasses import dataclass

@dataclass
class ChunkerConfig:
    """Configuration for text chunking"""
    
    # Chunk size parameters
    base_chunk_size: int = 500
    min_chunk_size: int = 100
    chunk_overlap: int = 50
    
    # Safety parameters
    max_chunks: int = 1000  # Prevent infinite loops or excessive chunking


