"""Configuration for vector databases."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class VectorDBConfig:
    """Base configuration for vector databases"""
    collection_name: str = "documents"
    embedding_dimension: int = 768  # Default for many models, adjust based on your model
    persist_directory: Optional[str] = None
    distance_metric: str = "cosine"  # "cosine", "euclidean", "dot"

@dataclass
class ChromaConfig(VectorDBConfig):
    """Configuration for ChromaDB"""
    persist_directory: str = "./chroma_db"
    client_settings: Dict[str, Any] = field(default_factory=dict)


