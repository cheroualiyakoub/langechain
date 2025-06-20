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
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")



class EmbeddingError(Exception):
    """Exception raised for errors in the embedding process"""
    pass
