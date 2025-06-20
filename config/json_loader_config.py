"""Configuration settings for patent JSON loaders"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Calculate project root based on config file location
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
DEFAULT_PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
DEFAULT_EPO_DIR = DEFAULT_PARSED_DIR / "EPO"
DEFAULT_USPTO_DIR = DEFAULT_PARSED_DIR / "USPTO"

@dataclass
class PatentLoaderConfig:
    """Configuration for patent document loaders"""
    # Paths
    project_root: Path = PROJECT_ROOT
    parsed_dir: Path = DEFAULT_PARSED_DIR
    epo_dir: Path = DEFAULT_EPO_DIR
    uspto_dir: Path = DEFAULT_USPTO_DIR
    
    # Processing options
    recursive: bool = True
    file_extension: str = ".json"
    max_files: Optional[int] = None
    encoding: str = 'utf-8'
    
    # Content extraction options
    include_abstract: bool = True
    include_description: bool = True
    include_claims: bool = True
    
    # Debug options
    verbose: bool = False
    
    def __post_init__(self):
        """Convert string paths to Path objects if needed"""
        for field in ['project_root', 'parsed_dir', 'epo_dir', 'uspto_dir']:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, Path(value))