"""
EPO Raw Data File Path Collection and JSON Parsing
Functions to collect all file paths from extracted EPO data and parse them to JSON.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_all_epo_file_paths(raw_data_directory: str = "../data/raw/EPO") -> Dict[str, List[str]]:
    """
    Collect all file paths from the raw EPO data directory.
    
    Args:
        raw_data_directory: Path to the raw EPO data directory
        
    Returns:
        Dictionary with categorized file paths:
        {
            'xml_files': [list of XML file paths],
            'pdf_files': [list of PDF file paths],
            'all_files': [list of all file paths],
            'archives': [list of archive names],
            'summary': {statistics}
        }
    """
    raw_data_path = Path(raw_data_directory)
    
    if not raw_data_path.exists():
        logger.error(f"Raw data directory not found: {raw_data_path}")
        return {
            'xml_files': [],
            'pdf_files': [],
            'all_files': [],
            'archives': [],
            'summary': {'error': 'Directory not found'}
        }
    
    # Initialize collections
    xml_files = []
    pdf_files = []
    all_files = []
    archives = []
    
    # Statistics
    stats = {
        'total_files': 0,
        'xml_count': 0,
        'pdf_count': 0,
        'archive_count': 0,
        'subfolder_count': 0,
        'zip_folder_count': 0
    }
    
    logger.info(f"Collecting file paths from: {raw_data_path}")
    
    # Loop through all archive directories
    for archive_item in raw_data_path.iterdir():
        if not archive_item.is_dir() or archive_item.name.startswith('.'):
            continue
            
        if archive_item.name.startswith('EPRTBJV'):
            archives.append(archive_item.name)
            stats['archive_count'] += 1
            
            logger.info(f"Processing archive: {archive_item.name}")
            
            # Loop through subfolders in archive (e.g., EPNWA1, EPNWB1, etc.)
            for subfolder in archive_item.iterdir():
                if not subfolder.is_dir():
                    continue
                    
                stats['subfolder_count'] += 1
                logger.debug(f"  Processing subfolder: {subfolder.name}")
                
                # Loop through ZIP folders in subfolder
                for zip_folder in subfolder.iterdir():
                    if not zip_folder.is_dir():
                        continue
                        
                    stats['zip_folder_count'] += 1
                    
                    # Loop through files in ZIP folder
                    for file_path in zip_folder.iterdir():
                        if file_path.is_file():
                            file_path_str = str(file_path)
                            all_files.append(file_path_str)
                            stats['total_files'] += 1
                            
                            # Categorize by file type
                            if file_path.suffix.lower() == '.xml':
                                xml_files.append(file_path_str)
                                stats['xml_count'] += 1
                            elif file_path.suffix.lower() == '.pdf':
                                pdf_files.append(file_path_str)
                                stats['pdf_count'] += 1
    
    # Create summary
    stats['archives_processed'] = len(archives)
    
    result = {
        'xml_files': xml_files,
        'pdf_files': pdf_files,
        'all_files': all_files,
        'archives': archives,
        'summary': stats
    }
    
    logger.info(f"File collection completed:")
    logger.info(f"  Archives: {stats['archive_count']}")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  XML files: {stats['xml_count']}")
    logger.info(f"  PDF files: {stats['pdf_count']}")
    
    return result


def get_file_paths_by_archive(raw_data_directory: str = "../data/raw/EPO") -> Dict[str, Dict]:
    """
    Get file paths organized by archive name.
    
    Args:
        raw_data_directory: Path to the raw EPO data directory
        
    Returns:
        Dictionary organized by archive:
        {
            'archive_name': {
                'xml_files': [list of XML paths],
                'pdf_files': [list of PDF paths],
                'subfolders': {
                    'subfolder_name': {
                        'xml_files': [...],
                        'pdf_files': [...],
                        'zip_folders': [...]
                    }
                }
            }
        }
    """
    raw_data_path = Path(raw_data_directory)
    
    if not raw_data_path.exists():
        logger.error(f"Raw data directory not found: {raw_data_path}")
        return {}
    
    archives_data = {}
    
    # Loop through all archive directories
    for archive_item in raw_data_path.iterdir():
        if not archive_item.is_dir() or not archive_item.name.startswith('EPRTBJV'):
            continue
            
        archive_name = archive_item.name
        archives_data[archive_name] = {
            'xml_files': [],
            'pdf_files': [],
            'subfolders': {}
        }
        
        logger.info(f"Processing archive: {archive_name}")
        
        # Loop through subfolders
        for subfolder in archive_item.iterdir():
            if not subfolder.is_dir():
                continue
                
            subfolder_name = subfolder.name
            archives_data[archive_name]['subfolders'][subfolder_name] = {
                'xml_files': [],
                'pdf_files': [],
                'zip_folders': []
            }
            
            # Loop through ZIP folders
            for zip_folder in subfolder.iterdir():
                if not zip_folder.is_dir():
                    continue
                    
                zip_folder_name = zip_folder.name
                archives_data[archive_name]['subfolders'][subfolder_name]['zip_folders'].append(zip_folder_name)
                
                # Collect files in this ZIP folder
                for file_path in zip_folder.iterdir():
                    if file_path.is_file():
                        file_path_str = str(file_path)
                        
                        if file_path.suffix.lower() == '.xml':
                            archives_data[archive_name]['xml_files'].append(file_path_str)
                            archives_data[archive_name]['subfolders'][subfolder_name]['xml_files'].append(file_path_str)
                        elif file_path.suffix.lower() == '.pdf':
                            archives_data[archive_name]['pdf_files'].append(file_path_str)
                            archives_data[archive_name]['subfolders'][subfolder_name]['pdf_files'].append(file_path_str)
    
    return archives_data


def filter_files_by_type(file_paths: List[str], file_type: str) -> List[str]:
    """
    Filter file paths by file type.
    
    Args:
        file_paths: List of file paths
        file_type: File type to filter ('xml', 'pdf', or extension like '.xml')
        
    Returns:
        List of filtered file paths
    """
    if not file_type.startswith('.'):
        file_type = f'.{file_type.lower()}'
    
    return [path for path in file_paths if Path(path).suffix.lower() == file_type]


def get_patent_files_only(file_paths: List[str]) -> List[str]:
    """
    Filter out TOC.xml files and return only patent files.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of patent file paths (excluding TOC.xml)
    """
    patent_files = []
    
    for path in file_paths:
        file_path = Path(path)
        filename = file_path.name.lower()
        
        # Include XML files that are not TOC.xml
        if file_path.suffix.lower() == '.xml' and not filename.startswith('toc.'):
            patent_files.append(path)
        # Include all PDF files
        elif file_path.suffix.lower() == '.pdf':
            patent_files.append(path)
    
    return patent_files


def save_file_paths_to_json(file_paths_data: Dict, output_file: str) -> bool:
    """
    Save file paths data to a JSON file.
    
    Args:
        file_paths_data: Dictionary with file paths data
        output_file: Path to output JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_paths_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"File paths saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving file paths to JSON: {e}")
        return False


def print_file_collection_summary(file_paths_data: Dict) -> None:
    """
    Print a summary of collected file paths.
    
    Args:
        file_paths_data: Dictionary with file paths data from collect_all_epo_file_paths
    """
    print("📊 EPO File Collection Summary")
    print("=" * 50)
    
    summary = file_paths_data.get('summary', {})
    
    print(f"📁 Archives processed: {summary.get('archive_count', 0)}")
    print(f"📂 Subfolders processed: {summary.get('subfolder_count', 0)}")
    print(f"📦 ZIP folders processed: {summary.get('zip_folder_count', 0)}")
    print(f"\n📄 Total files: {summary.get('total_files', 0)}")
    print(f"📋 XML files: {summary.get('xml_count', 0)}")
    print(f"📑 PDF files: {summary.get('pdf_count', 0)}")
    
    print(f"\n📦 Archives found:")
    for archive in file_paths_data.get('archives', []):
        print(f"  • {archive}")
    
    # Show sample file paths
    xml_files = file_paths_data.get('xml_files', [])
    pdf_files = file_paths_data.get('pdf_files', [])
    
    if xml_files:
        print(f"\n📋 Sample XML files:")
        for xml_file in xml_files[:3]:
            print(f"  • {Path(xml_file).name}")
        if len(xml_files) > 3:
            print(f"  ... and {len(xml_files) - 3} more XML files")
    
    if pdf_files:
        print(f"\n📑 Sample PDF files:")
        for pdf_file in pdf_files[:3]:
            print(f"  • {Path(pdf_file).name}")
        if len(pdf_files) > 3:
            print(f"  ... and {len(pdf_files) - 3} more PDF files")


def get_all_epo_file_paths(raw_data_directory: str = "../data/raw/EPO") -> List[str]:
    """
    Simple function to get all file paths from the raw EPO data directory.
    
    Args:
        raw_data_directory: Path to the raw EPO data directory
        
    Returns:
        List of all file paths as strings
        
    Example:
        >>> paths = get_all_epo_file_paths()
        >>> print(paths[0])
        data/raw/EPO/EPRTBJV2025000021001001/EPNWA1/EP22741291NWA1/EP22741291NWA1.xml
    """
    raw_data_path = Path(raw_data_directory)
    file_paths = []
    
    if not raw_data_path.exists():
        print(f"❌ Directory not found: {raw_data_path}")
        return file_paths
    
    # Recursively find all files
    for file_path in raw_data_path.rglob("*"):
        if file_path.is_file():
            file_paths.append(str(file_path))
    
    return file_paths


def get_epo_xml_files(raw_data_directory: str = "../data/raw/EPO") -> List[str]:
    """
    Get only XML file paths from the raw EPO data directory.
    
    Args:
        raw_data_directory: Path to the raw EPO data directory
        
    Returns:
        List of XML file paths as strings
    """
    all_files = get_all_epo_file_paths(raw_data_directory)
    return [f for f in all_files if f.lower().endswith('.xml')]


def get_epo_pdf_files(raw_data_directory: str = "../data/raw/EPO") -> List[str]:
    """
    Get only PDF file paths from the raw EPO data directory.
    
    Args:
        raw_data_directory: Path to the raw EPO data directory
        
    Returns:
        List of PDF file paths as strings
    """
    all_files = get_all_epo_file_paths(raw_data_directory)
    return [f for f in all_files if f.lower().endswith('.pdf')]


def get_epo_patent_files(raw_data_directory: str = "../data/raw/EPO") -> List[str]:
    """
    Get only patent files (XML and PDF, but exclude TOC.xml) from the raw EPO data directory.
    
    Args:
        raw_data_directory: Path to the raw EPO data directory
        
    Returns:
        List of patent file paths as strings (no TOC.xml files)
    """
    all_files = get_all_epo_file_paths(raw_data_directory)
    patent_files = []
    
    for file_path in all_files:
        filename = Path(file_path).name.lower()
        if filename.endswith('.xml') and not filename.startswith('toc.'):
            patent_files.append(file_path)
        elif filename.endswith('.pdf'):
            patent_files.append(file_path)
    
    return patent_files

# Example usage
if __name__ == "__main__":
    # Collect all file paths
    file_paths = collect_all_epo_file_paths()
    
    # Print summary
    print_file_collection_summary(file_paths)
    
    # Save to JSON
    save_file_paths_to_json(file_paths, "../data/processed/epo_file_paths.json")
    
    # Get patent files only (excluding TOC.xml)
    patent_files = get_patent_files_only(file_paths['all_files'])
    print(f"\n🎯 Patent files only: {len(patent_files)} files")