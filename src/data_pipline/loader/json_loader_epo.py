import os
import json
import sys
from typing import List, Dict
from pathlib import Path
from langchain.schema import Document

# Calculate the path to config directory
project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
config_dir = project_root / "config"

# Add config directory to Python path
if str(config_dir) not in sys.path:
    sys.path.append(str(config_dir))

# Now import from data_config
try:
    from data_config import RAW_JSON_DIR, PARCED_DATA_DIR
    print("✅ Successfully imported from data_config.py")
except ImportError as e:
    print(f"⚠️ Couldn't import from data_config.py: {e}")
    print(f"⚠️ Using default paths instead")
    RAW_JSON_DIR = str(project_root / "data" / "parced" / "EPO")
    PARCED_DATA_DIR = str(project_root / "data" / "parced")



def get_epo_json_file_paths(base_dir: str = None) -> List[str]:
    """
    Get all EPO JSON file paths from the data/parced/EPO folder recursively.
    
    Args:
        base_dir: Base directory path (optional, defaults to project root)
        
    Returns:
        List of string paths to all EPO JSON files
    """
    if base_dir is None:
        # Default to project root (adjust path levels as needed)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent  # Go up to project root
        epo_parced_dir = project_root / "data" / "parced" / "EPO"
    else:
        epo_parced_dir = Path(base_dir) / "data" / "parced" / "EPO"
         
    json_file_paths = []
    
    if epo_parced_dir.exists():
        # Recursively find all JSON files in EPO directory
        for json_file in epo_parced_dir.rglob("*.json"):
            if json_file.is_file():
                json_file_paths.append(str(json_file))
    else:
        print(f"⚠️ Warning: EPO parced directory does not exist: {epo_parced_dir}")
    
    print(f"📁 Found {len(json_file_paths)} EPO JSON files")
    return json_file_paths


def get_all_json_file_paths(base_dir: str = None) -> List[str]:
    """
    Get all JSON file paths from the data/parced folder recursively.
    
    Args:
        base_dir: Base directory path (optional, defaults to project root)
        
    Returns:
        List of string paths to all JSON files in parced directory
    """
    if base_dir is None:
        # Use the PARCED_DATA_DIR from config
        parced_dir = Path(PARCED_DATA_DIR)
    else:
        parced_dir = Path(base_dir) / "data" / "parced"
    
    json_file_paths = []
    
    if parced_dir.exists():
        # Recursively find all JSON files
        for json_file in parced_dir.rglob("*.json"):
            if json_file.is_file():
                json_file_paths.append(str(json_file))
    else:
        print(f"⚠️ Warning: Parced directory does not exist: {parced_dir}")
    
    print(f"📁 Found {len(json_file_paths)} JSON files in parced directory")
    return json_file_paths


def load_json_documents() -> List[Document]:
    """Load JSON documents from the RAW_JSON_DIR (parced EPO directory)."""
    docs = []
    
    if not os.path.exists(RAW_JSON_DIR):
        print(f"⚠️ Warning: Directory does not exist: {RAW_JSON_DIR}")
        return docs
    
    # Get all JSON files recursively from RAW_JSON_DIR
    json_dir_path = Path(RAW_JSON_DIR)
    for json_file in json_dir_path.rglob("*.json"):
        try:
            with open(json_file, "r", encoding='utf-8') as f:
                data = json.load(f)

                # Extract content based on your JSON structure
                content_parts = []
                
                # Add abstract if available
                if data.get("bibliographic_data", {}).get("abstract"):
                    content_parts.append(data["bibliographic_data"]["abstract"])
                
                # Add main sections content
                for section in data.get("main_sections", []):
                    section_text = f"## {section.get('heading_text', '')}\n"
                    for para in section.get("paragraphs", []):
                        section_text += para.get("text", "") + "\n"
                    content_parts.append(section_text)
                
                # Add claims
                claims_text = "## Claims\n"
                for claim in data.get("claims", []):
                    claims_text += f"Claim {claim.get('claim_number', '')}: {claim.get('text', '')}\n"
                if data.get("claims"):
                    content_parts.append(claims_text)
                
                # Combine all content or fallback to JSON dump
                content = "\n\n".join(content_parts) if content_parts else json.dumps(data)
                
                # Create metadata
                metadata = {
                    "filename": json_file.name,
                    "filepath": str(json_file),
                    "patent_id": data.get("bibliographic_data", {}).get("publication_reference", {}).get("document_id", {}).get("doc_number"),
                    "title": data.get("bibliographic_data", {}).get("invention_title", {}).get("text"),
                    "applicant": data.get("bibliographic_data", {}).get("parties", {}).get("applicants", [{}])[0].get("addressbook", {}).get("name", {}).get("text") if data.get("bibliographic_data", {}).get("parties", {}).get("applicants") else None,
                    "publication_date": data.get("bibliographic_data", {}).get("publication_reference", {}).get("document_id", {}).get("date")
                }

                docs.append(Document(page_content=content, metadata=metadata))
                
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
            continue
    
    print(f"📄 Loaded {len(docs)} documents from JSON files")
    return docs

# Usage examples:
if __name__ == "__main__":
    print(f"📂 Using RAW_JSON_DIR: {RAW_JSON_DIR}")
    print(f"📂 Using PARCED_DATA_DIR: {PARCED_DATA_DIR}")
    
    # Get all JSON files from parced directory
    all_json_files = get_all_json_file_paths()
    print(f"Total JSON files: {len(all_json_files)}")
    
    # Get only EPO JSON files
    epo_json_files = get_epo_json_file_paths()
    print(f"EPO JSON files: {len(epo_json_files)}")
    
    # Print first few files as examples
    if epo_json_files:
        print("\nSample EPO JSON files:")
        for i, file_path in enumerate(epo_json_files[:3]):
            print(f"  {i+1}. {file_path}")
    
    # Load documents
    documents = load_json_documents()
    print(f"\nLoaded {len(documents)} documents")