"""
EPO Archive Data Extraction Pipeline
Processes EPO patent archives from EUPROP data and extracts only patent XML/PDF files.
Excludes TOC.xml files and stores files directly without keeping unzipped folders.
"""

import os
import zipfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedEpoExtractor:
    """
    Optimized EPO patent extractor that processes all archives and stores only patent files.
    
    Features:
    - Handles both ZIP files and extracted directories
    - Processes all EPO archives in the directory
    - Extracts only patent XML and PDF files (excludes TOC.xml)
    - Stores files directly without temporary folder structure
    - Memory efficient with minimal disk usage
    """
    
    def __init__(self, base_archive_directory: str, base_output_directory: str):
        """
        Initialize the optimized EPO extractor.
        
        Args:
            base_archive_directory: Path to EPO archive directory (e.g., '../data/archive/EPO')
            base_output_directory: Path to raw data output directory (e.g., '../data/raw/EPO')
        """
        self.base_archive_dir = Path(base_archive_directory)
        self.base_output_dir = Path(base_output_directory)
        
        # Create output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.global_stats = {
            'archives_found': 0,
            'archives_processed': 0,
            'zip_files_processed': 0,
            'xml_files_extracted': 0,
            'pdf_files_extracted': 0,
            'toc_files_skipped': 0,
            'other_files_skipped': 0,
            'errors': 0
        }
    
    def find_all_epo_archives(self) -> List[Dict]:
        """
        Find all EPO archives (both ZIP files and directories) in the base directory.
        
        Returns:
            List of archive information dictionaries
        """
        archives = []
        
        if not self.base_archive_dir.exists():
            logger.error(f"EPO archive directory not found: {self.base_archive_dir}")
            return archives
        
        for item in self.base_archive_dir.iterdir():
            if item.name.startswith('.'):  # Skip hidden files
                continue
                
            if item.is_file() and item.suffix.lower() == '.zip' and item.name.startswith('EPRTBJV'):
                # Main archive ZIP file
                archives.append({
                    'name': item.stem,
                    'path': item,
                    'type': 'main_zip'
                })
            elif item.is_dir() and item.name.startswith('EPRTBJV'):
                # Already extracted archive directory
                archives.append({
                    'name': item.name,
                    'path': item,
                    'type': 'directory'
                })
        
        self.global_stats['archives_found'] = len(archives)
        logger.info(f"Found {len(archives)} EPO archives")
        
        return archives
    
    def is_patent_file(self, filename: str) -> bool:
        """
        Check if a file is a patent XML or PDF file (excludes TOC.xml).
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if it's a patent file we want to keep
        """
        filename_lower = filename.lower()
        
        # Check for patent XML files (exclude TOC.xml)
        if filename_lower.endswith('.xml'):
            return not filename_lower.startswith('toc.')
        
        # Check for PDF files
        if filename_lower.endswith('.pdf'):
            return True
            
        return False
    
    def extract_from_zip_file(self, zip_file_path: Path, output_dir: Path) -> Dict:
        """
        Extract patent files directly from a ZIP file.
        
        Args:
            zip_file_path: Path to the ZIP file
            output_dir: Output directory for extracted files
            
        Returns:
            Statistics dictionary for this ZIP file
        """
        stats = {
            'xml_extracted': 0,
            'pdf_extracted': 0,
            'toc_skipped': 0,
            'other_skipped': 0,
            'errors': 0
        }
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                for file_name in file_list:
                    try:
                        # Skip directories
                        if file_name.endswith('/'):
                            continue
                        
                        # Get just the filename (no path)
                        base_filename = os.path.basename(file_name)
                        
                        if self.is_patent_file(base_filename):
                            # Extract patent file directly
                            output_file = output_dir / base_filename
                            
                            # Extract file data
                            file_data = zip_ref.read(file_name)
                            
                            # Write directly to output
                            with open(output_file, 'wb') as f:
                                f.write(file_data)
                            
                            # Update stats
                            if base_filename.lower().endswith('.xml'):
                                stats['xml_extracted'] += 1
                            elif base_filename.lower().endswith('.pdf'):
                                stats['pdf_extracted'] += 1
                                
                        elif base_filename.lower().startswith('toc.'):
                            stats['toc_skipped'] += 1
                        else:
                            stats['other_skipped'] += 1
                            
                    except Exception as e:
                        logger.warning(f"Error extracting {file_name}: {e}")
                        stats['errors'] += 1
                        
        except Exception as e:
            logger.error(f"Error processing ZIP file {zip_file_path.name}: {e}")
            stats['errors'] += 1
            
        return stats
    
    def process_doc_folder(self, doc_folder: Path, archive_output_dir: Path) -> Dict:
        """
        Process a DOC folder containing ZIP files with patent data.
        
        Args:
            doc_folder: Path to the DOC folder
            archive_output_dir: Output directory for this archive
            
        Returns:
            Statistics dictionary for this DOC folder
        """
        stats = {
            'zip_files_processed': 0,
            'xml_files_extracted': 0,
            'pdf_files_extracted': 0,
            'toc_files_skipped': 0,
            'other_files_skipped': 0,
            'errors': 0
        }
        
        # Process each subfolder in DOC
        subfolders = [sf for sf in doc_folder.iterdir() if sf.is_dir()]
        logger.info(f"Processing {len(subfolders)} subfolders in DOC")
        
        for subfolder in subfolders:
            subfolder_output = archive_output_dir / subfolder.name
            subfolder_output.mkdir(parents=True, exist_ok=True)
            
            zip_files = list(subfolder.glob("*.zip"))
            if not zip_files:
                continue
                
            logger.info(f"Processing {subfolder.name}: {len(zip_files)} ZIP files")
            
            for i, zip_file in enumerate(zip_files, 1):
                if len(zip_files) > 100 and i % 100 == 0:
                    logger.info(f"  Progress: {i}/{len(zip_files)} ZIP files")
                
                # Create individual folder for this ZIP's content
                zip_output_dir = subfolder_output / zip_file.stem
                zip_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract patent files from this ZIP
                zip_stats = self.extract_from_zip_file(zip_file, zip_output_dir)
                
                # Aggregate statistics
                stats['zip_files_processed'] += 1
                stats['xml_files_extracted'] += zip_stats['xml_extracted']
                stats['pdf_files_extracted'] += zip_stats['pdf_extracted']
                stats['toc_files_skipped'] += zip_stats['toc_skipped']
                stats['other_files_skipped'] += zip_stats['other_skipped']
                stats['errors'] += zip_stats['errors']
        
        return stats
    
    def process_main_archive_zip(self, archive_info: Dict) -> Dict:
        """
        Process a main archive ZIP file (like EPRTBJV2025000024001001.zip).
        
        Args:
            archive_info: Dictionary with archive information
            
        Returns:
            Statistics dictionary for this archive
        """
        archive_path = archive_info['path']
        archive_name = archive_info['name']
        
        logger.info(f"Processing main ZIP archive: {archive_name}")
        
        # Create output directory for this archive
        archive_output_dir = self.base_output_dir / archive_name
        archive_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use temporary directory for main ZIP extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract main ZIP file
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                
                # Find DOC folder
                doc_folder = None
                for item in temp_path.rglob('*'):
                    if item.is_dir() and item.name.upper() in ['DOC', 'DOCS']:
                        doc_folder = item
                        break
                
                if not doc_folder:
                    logger.error(f"No DOC folder found in {archive_name}")
                    return {'errors': 1}
                
                # Process the DOC folder
                return self.process_doc_folder(doc_folder, archive_output_dir)
                
        except Exception as e:
            logger.error(f"Error processing main archive {archive_name}: {e}")
            return {'errors': 1}
    
    def process_extracted_archive_directory(self, archive_info: Dict) -> Dict:
        """
        Process an already extracted archive directory.
        
        Args:
            archive_info: Dictionary with archive information
            
        Returns:
            Statistics dictionary for this archive
        """
        archive_path = archive_info['path']
        archive_name = archive_info['name']
        
        logger.info(f"Processing extracted directory: {archive_name}")
        
        # Create output directory for this archive
        archive_output_dir = self.base_output_dir / archive_name
        archive_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find DOC folder
        doc_folder = archive_path / "DOC"
        if not doc_folder.exists():
            logger.error(f"No DOC folder found in {archive_name}")
            return {'errors': 1}
        
        # Process the DOC folder
        return self.process_doc_folder(doc_folder, archive_output_dir)
    
    def extract_all_archives(self, verbose: bool = True) -> Dict:
        """
        Extract all EPO archives in the base directory.
        
        Args:
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with global processing statistics
        """
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Starting Optimized EPO Archive Extraction")
            print(f"📂 Archive directory: {self.base_archive_dir}")
            print(f"📁 Output directory: {self.base_output_dir}")
            print(f"🎯 Target: Patent XML & PDF files only (no TOC.xml)")
            print("=" * 60)
        
        # Find all archives
        archives = self.find_all_epo_archives()
        
        if not archives:
            logger.warning("No EPO archives found")
            return self.global_stats
        
        if verbose:
            print(f"📦 Found {len(archives)} EPO archives:")
            for archive in archives:
                print(f"  • {archive['name']} ({archive['type']})")
        
        # Process each archive
        for i, archive in enumerate(archives, 1):
            if verbose:
                print(f"\n[{i}/{len(archives)}] Processing: {archive['name']}")
            
            try:
                if archive['type'] == 'main_zip':
                    result = self.process_main_archive_zip(archive)
                else:
                    result = self.process_extracted_archive_directory(archive)
                
                # Aggregate global statistics
                if result:
                    self.global_stats['archives_processed'] += 1
                    self.global_stats['zip_files_processed'] += result.get('zip_files_processed', 0)
                    self.global_stats['xml_files_extracted'] += result.get('xml_files_extracted', 0)
                    self.global_stats['pdf_files_extracted'] += result.get('pdf_files_extracted', 0)
                    self.global_stats['toc_files_skipped'] += result.get('toc_files_skipped', 0)
                    self.global_stats['other_files_skipped'] += result.get('other_files_skipped', 0)
                    self.global_stats['errors'] += result.get('errors', 0)
                    
                    if verbose:
                        print(f"  ✅ {archive['name']}: {result.get('xml_files_extracted', 0)} XML, "
                              f"{result.get('pdf_files_extracted', 0)} PDF extracted")
                else:
                    self.global_stats['errors'] += 1
                    if verbose:
                        print(f"  ❌ Failed to process {archive['name']}")
                        
            except Exception as e:
                logger.error(f"Error processing archive {archive['name']}: {e}")
                self.global_stats['errors'] += 1
                if verbose:
                    print(f"  ❌ Error processing {archive['name']}: {e}")
        
        # Final results
        end_time = time.time()
        total_time = end_time - start_time
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"🎉 EPO Archive Extraction Completed!")
            print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"\n📊 Global Statistics:")
            print(f"   Archives found: {self.global_stats['archives_found']}")
            print(f"   Archives processed: {self.global_stats['archives_processed']}")
            print(f"   ZIP files processed: {self.global_stats['zip_files_processed']}")
            print(f"   📄 XML files extracted: {self.global_stats['xml_files_extracted']}")
            print(f"   📑 PDF files extracted: {self.global_stats['pdf_files_extracted']}")
            print(f"   📋 Total patent files: {self.global_stats['xml_files_extracted'] + self.global_stats['pdf_files_extracted']}")
            print(f"   ⏩ TOC files skipped: {self.global_stats['toc_files_skipped']}")
            print(f"   🗑️ Other files skipped: {self.global_stats['other_files_skipped']}")
            print(f"   ❌ Errors: {self.global_stats['errors']}")
            
            if total_time > 60:
                rate = self.global_stats['zip_files_processed'] / (total_time / 60)
                print(f"   📈 Processing rate: {rate:.1f} ZIP files/minute")
        
        return self.global_stats


def extract_epo_archives(archive_directory: str, 
                        output_directory: str,
                        verbose: bool = True) -> Dict:
    """
    Main function to extract all EPO archives with optimized patent file extraction.
    
    Args:
        archive_directory: Path to EPO archive directory (e.g., '../data/archive/EPO')
        output_directory: Path to raw data output directory (e.g., '../data/raw/EPO')
        verbose: Whether to print detailed progress information (default: True)
        
    Returns:
        Dictionary with processing statistics
        
    Features:
        - Processes all EPO archives (ZIP files and directories)
        - Extracts only patent XML and PDF files
        - Excludes TOC.xml files
        - Stores files directly without keeping extracted folder structure
        - Memory efficient processing
        
    Example:
        >>> result = extract_epo_archives(
        ...     archive_directory='../data/archive/EPO',
        ...     output_directory='../data/raw/EPO',
        ...     verbose=True
        ... )
        >>> print(f"Extracted {result['xml_files_extracted']} XML and {result['pdf_files_extracted']} PDF files")
    """
    extractor = OptimizedEpoExtractor(
        base_archive_directory=archive_directory,
        base_output_directory=output_directory
    )
    
    return extractor.extract_all_archives(verbose=verbose)


# Maintain backward compatibility
def extract_epo_archives_old(archive_directory: str, 
                            output_directory: str,
                            file_types: List[str] = ['.xml', '.pdf'],
                            preserve_hierarchy: bool = True,
                            verbose: bool = True,
                            temp_dir: str = None) -> Dict:
    """
    Backward compatibility wrapper for the old function signature.
    Now uses the optimized extractor.
    """
    return extract_epo_archives(
        archive_directory=archive_directory,
        output_directory=output_directory,
        verbose=verbose
    )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    archive_dir = "../data/archive/EPO"
    raw_data_dir = "../data/raw/EPO"
    
    # Run optimized extraction
    result = extract_epo_archives(
        archive_directory=archive_dir,
        output_directory=raw_data_dir,
        verbose=True
    )
    
    print(f"\n🎉 Extraction completed!")
    print(f"📊 Final statistics: {result}")