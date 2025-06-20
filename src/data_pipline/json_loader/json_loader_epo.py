"""Specialized loader for EPO (European Patent Office) patent JSON files"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

from langchain.schema import Document

from .json_loader_patent import PatentJSONLoader, logger
from json_loader_config import PatentLoaderConfig


class EPOPatentLoader(PatentJSONLoader):
    """Loader for EPO patent JSON files"""
    
    def get_default_directory(self) -> Path:
        """Return the default EPO directory from config"""
        return self.config.epo_dir
    
    # def extract_content(self, data: Dict[str, Any]) -> str:
    #     """Extract content from EPO patent JSON data following extract_documents pattern
        
    #     Args:
    #         data: JSON patent data
            
    #     Returns:
    #         Extracted text content with clear section headers
    #     """
    #     content_parts = []
        
    #     # Add title if available (preferring English version)
    #     biblio = data.get("bibliographic_data", {})
    #     title_dict = biblio.get("title", {})
        
    #     if isinstance(title_dict, dict):
    #         title = title_dict.get("en") or next(iter(title_dict.values()), "")
    #     else:
    #         # Fallback for different structure
    #         title = biblio.get("invention_title", {}).get("text", "")
            
    #     if title:
    #         content_parts.append(f"# TITLE\n{title}")
        
    #     # Add abstract if available and configured
    #     abstract = biblio.get("abstract")
    #     if self.config.include_abstract and abstract:
    #         content_parts.append(f"# ABSTRACT\n{abstract}")
        
    #     # Add claims if configured
    #     if self.config.include_claims:
    #         for claim in data.get("claims", []):
    #             claim_num = claim.get("claim_number", "")
    #             claim_text = claim.get("text", "")
    #             content_parts.append(f"# CLAIM {claim_num}\n{claim_text}")
        
    #     # Add main sections if configured
    #     if self.config.include_description:
    #         for section in data.get("main_sections", []):
    #             section_name = section.get("heading_text", "UNKNOWN_SECTION")
                
    #             for para in section.get("paragraphs", []):
    #                 para_text = para.get("text", "")
    #                 para_id = para.get("p_id", "")
    #                 content_parts.append(f"# {section_name} [P_ID: {para_id}]\n{para_text}")
        
    #     # Combine all content or fallback to JSON dump
    #     content = "\n\n".join(content_parts) if content_parts else json.dumps(data)
    #     return content

    def extract_title(self, data: Dict[str, Any]) -> Dict[str, Any]:
            # Title (en preferred)
        """Extract title from EPO patent JSON data following extract_documents pattern
        Args:
            data: JSON patent data
        Returns:
            Dictionary with title
        """
        biblio = data.get("bibliographic_data", {})

        title_dict = biblio.get("title", {})
        title = title_dict.get("en") or next(iter(title_dict.values()), "")
        if title:
            return {"title": title}
        
        return {}
    
    # def extract_abstract(self, data: Dict[str, Any]) -> Dict[str, Any]:
    #     """Extract abstract from EPO patent JSON data following extract_documents pattern
        
    #     Args:
    #         data: JSON patent data
            
    #     Returns:
    #         Dictionary with abstract
    #     """
    #     biblio = data.get("bibliographic_data", {})
    #     abstract = biblio.get("abstract")
        
    #     if abstract:
    #         return {"abstract": abstract}
    #     return {}

    # def extract_claims(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #     """Extract claims from EPO patent JSON data following extract_documents pattern
        
    #     Args:
    #         data: JSON patent data
            
    #     Returns:
    #         List of dictionaries with claims
    #     """
    #     claims = []
    #     for claim in data.get("claims", []):
    #         claim_num = claim.get("claim_number", "")
    #         claim_text = claim.get("text", "")
    #         claims.append({
    #             "claim_number": claim_num,
    #             "text": claim_text
    #         })

    #             # Claims
    #     for claim in json_data.get("claims", []):
    #         documents.append(Document(
    #             page_content=claim["text"],
    #             metadata={**common_meta, "section": "claim", "claim_number": claim.get("claim_number")}
    #         ))
    #     return claims
    
    def extract_metadata_content(self, data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Extract metadata from EPO patent JSON data matching extract_documents pattern
        
        Args:
            data: JSON patent data
            file_path: Path to the JSON file
            
        Returns:
            Dictionary of metadata
        """
        biblio = data.get("bibliographic_data", {})
        
        documents = []

            # Fix the incompatible metadata types
        ipc_classes = biblio.get("ipc_classes", [])
        ipc_classes_str = ", ".join(ipc_classes) if isinstance(ipc_classes, list) else str(ipc_classes)

        common_meta = {
            "doc_id": biblio.get("doc_id", "UNKNOWN"),
            "language": biblio.get("language"),
            "country": biblio.get("country"),
            "doc_number": biblio.get("doc_number"),
            "application_number": biblio.get("application_number"),
            "publication_date": biblio.get("publication_date"),
            "ipc_classes": ipc_classes_str,
            "file":biblio.get("file"),
            "filePath":str(file_path.absolute()),
            "title": self.extract_title(data).get("title", ""),
        }

        title_dict = biblio.get("title", {})
        title = title_dict.get("en") or next(iter(title_dict.values()), "")
        if title:
            documents.append(Document(
                page_content=title,
                metadata={**common_meta, "section": "title"}
            ))
        
        abstract = biblio.get("abstract")
        if abstract:
            documents.append(Document(
                page_content=abstract,
                metadata={**common_meta, "section": "abstract"}
            ))
        
            # Main sections
        for section in data.get("main_sections", []):
            section_name = section.get("heading_text", "UNKNOWN_SECTION")
            for p in section.get("paragraphs", []):
                documents.append(Document(
                    page_content=f"{section_name}\n{p['text']}",
                    metadata={**common_meta, "section": section_name, "p_id": p.get("p_id")}
                ))

        for claim in data.get("claims", []):
            documents.append(Document(
                page_content=claim["text"],
                metadata={**common_meta, "section": "claim", "claim_number": claim.get("claim_number")}
            ))


        # Create comprehensive metadata matching the extract_documents function
        return documents
