"""Base classes for loading patent JSON files into LangChain documents"""

import os
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Generator, Union

from langchain.schema import Document

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "config"))
try:
    from json_loader_config import PatentLoaderConfig
except ImportError:
    raise ImportError("Could not import PatentLoaderConfig. Make sure json_loader_config.py is in the config directory.")

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PatentJSONLoader(ABC):
    """Base class for loading patent JSON files into LangChain documents"""
    
    def __init__(self, config: Optional[PatentLoaderConfig] = None):
        """Initialize the loader with configuration
        
        Args:
            config: Configuration for the loader (optional)
        """
        self.config = config or PatentLoaderConfig()
        logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")
        
    def get_json_files(self, directory: Optional[Union[str, Path]] = None) -> List[Path]:
        """Get all JSON files from a directory
        
        Args:
            directory: Directory to search (defaults to configured directory)
            
        Returns:
            List of Path objects for JSON files
        """
        search_dir = Path(directory) if directory else self.get_default_directory()
        
        if not search_dir.exists():
            logger.warning(f"Directory does not exist: {search_dir}")
            return []
        
        # Find JSON files
        if self.config.recursive:
            file_pattern = f"**/*{self.config.file_extension}"
        else:
            file_pattern = f"*{self.config.file_extension}"
            
        files = list(search_dir.glob(file_pattern))
        
        # Apply max_files limit if set
        if self.config.max_files and len(files) > self.config.max_files:
            files = files[:self.config.max_files]
            logger.info(f"Limited to {self.config.max_files} files out of {len(files)} total files")
            
        logger.info(f"Found {len(files)} JSON files in {search_dir}")
        return files

    def load_raw_json_files(self, directory: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Load raw JSON data from files without converting to Documents
        
        Args:
            directory: Directory to search (defaults to configured directory)
            
        Returns:
            List of raw JSON data dictionaries
        """
        search_dir = Path(directory) if directory else self.get_default_directory()
        files = self.get_json_files(search_dir)
        raw_data = []
        
        for i, file_path in enumerate(files):
            if self.config.verbose and i % 100 == 0:
                logger.info(f"Loading raw file {i+1}/{len(files)}: {file_path.name}")
            
            try:
                with open(file_path, "r", encoding=self.config.encoding) as f:
                    json_data = json.load(f)
                    raw_data.append({"file_path": str(file_path), "data": json_data})
            except Exception as e:
                logger.error(f"Error loading raw JSON from {file_path.name}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(raw_data)} raw JSON files")
        return raw_data
    
    @abstractmethod
    def get_default_directory(self) -> Path:
        """Return the default directory for this loader type"""
        pass
    
    @abstractmethod
    def extract_metadata_content(self, data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Extract metadata from patent JSON data
        
        Args:
            data: JSON patent data
            file_path: Path to the JSON file
            
        Returns:
            Dictionary of metadata
        """
        pass
    
    def load_single_document(self, file_path: Path) -> Optional[Document]:
        """Load a single document from a JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            LangChain Document or None if loading failed
        """
        try:
            with open(file_path, "r", encoding=self.config.encoding) as f:
                data = json.load(f)
            
            # content = self.extract_content(data)
            # metadata = self.extract_metadata(data, file_path)
            
            return self.extract_metadata_content(data, file_path)
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            return None
    
    def extract_document_components(self, directory: Optional[Union[str, Path]] = None) -> List[Document]:
        """Load all documents from the JSON files
        
        Args:
            directory: Directory to search (defaults to configured directory)
            
        Returns:
            List of LangChain Document objects
        """
        files = self.get_json_files(directory)
        documents = []
        
        for i, file_path in enumerate(files):
            if self.config.verbose and i % 100 == 0:
                logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
                
            doc = self.load_single_document(file_path)
            if doc:
                documents.append(doc)
        
        logger.info(f"Successfully loaded {len(documents)} documents from {len(files)} files")
        return documents
    
    def iter_documents(self, directory: Optional[Union[str, Path]] = None) -> Generator[Document, None, None]:
        """Iterator version of load_documents for memory efficiency
        
        Args:
            directory: Directory to search (defaults to configured directory)
            
        Yields:
            LangChain Document objects one at a time
        """
        files = self.get_json_files(directory)
        loaded = 0
        
        for i, file_path in enumerate(files):
            if self.config.verbose and i % 100 == 0:
                logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
                
            doc = self.load_single_document(file_path)
            if doc:
                loaded += 1
                yield doc
        
        logger.info(f"Successfully yielded {loaded} documents from {len(files)} files")