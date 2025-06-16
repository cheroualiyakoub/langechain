"""
Main Data Pipeline for Patent Data Processing
Orchestrates the extraction and processing of patent data from various sources.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add the EU_XML_data_loader to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
eu_loader_dir = src_dir / "load_raw_data_xml_pdf_EPO"
sys.path.append(str(eu_loader_dir))

from load_raw_data_xml_pdf_EPO import extract_epo_archives

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Main data pipeline for processing patent archives.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the data pipeline.
        
        Args:
            base_dir: Base directory path (defaults to project root)
        """
        if base_dir is None:
            # Default to project root (3 levels up from this file)
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Define standard paths
        self.data_dir = self.base_dir / "data"
        self.archive_dir = self.data_dir / "archive"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # EPO specific paths
        self.epo_archive_dir = self.archive_dir / "EPO"
        self.epo_raw_dir = self.raw_dir / "EPO"
        
        # USPTO specific paths  
        self.uspto_archive_dir = self.archive_dir / "USPTO"
        self.uspto_raw_dir = self.raw_dir / "USPTO"
        
        logger.info(f"Data pipeline initialized with base directory: {self.base_dir}")
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.archive_dir,
            self.raw_dir,
            self.processed_dir,
            self.epo_raw_dir,
            self.uspto_raw_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def check_epo_archive_structure(self) -> Dict:
        """
        Check the EPO archive structure and identify available archives.
        
        Returns:
            Dictionary with archive information
        """
        logger.info("Checking EPO archive structure...")
        
        if not self.epo_archive_dir.exists():
            logger.warning(f"EPO archive directory does not exist: {self.epo_archive_dir}")
            return {"archives": [], "total": 0}
        
        # Look for archive folders (like EPRTBJV2025000024001001)
        archive_folders = []
        for item in self.epo_archive_dir.iterdir():
            if item.is_dir():
                # Check if it has the expected structure (DOC folder)
                doc_folder = item / "DOC"
                if doc_folder.exists():
                    # Look for subfolders in DOC (like EPNWA1)
                    subfolders = [subfolder for subfolder in doc_folder.iterdir() if subfolder.is_dir()]
                    
                    archive_info = {
                        "name": item.name,
                        "path": item,
                        "doc_folder": doc_folder,
                        "subfolders": subfolders,
                        "zip_count": 0
                    }
                    
                    # Count ZIP files in subfolders
                    total_zips = 0
                    for subfolder in subfolders:
                        zip_files = list(subfolder.glob("*.zip"))
                        total_zips += len(zip_files)
                    
                    archive_info["zip_count"] = total_zips
                    archive_folders.append(archive_info)
                    
                    logger.info(f"Found archive: {item.name} with {total_zips} ZIP files")
        
        return {
            "archives": archive_folders,
            "total": len(archive_folders)
        }
    
    def extract_epo_data(self, verbose: bool = True) -> Dict:
        """
        Extract EPO patent data from all archives using the optimized extractor.
        
        Args:
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with extraction statistics
        """
        logger.info("Starting EPO data extraction...")
        
        # Ensure directories exist
        self.ensure_directories()
        
        if verbose:
            print(f"🚀 Starting EPO Data Extraction Pipeline")
            print(f"📂 Archive directory: {self.epo_archive_dir}")
            print(f"📁 Output directory: {self.epo_raw_dir}")
        
        # Use the optimized extractor to process all archives
        try:
            result = extract_epo_archives(
                archive_directory=str(self.epo_archive_dir),
                output_directory=str(self.epo_raw_dir),
                verbose=verbose
            )
            
            if verbose:
                print(f"\n✅ EPO extraction completed successfully!")
            
            return result
            
        except Exception as e:
            logger.error(f"EPO extraction failed: {str(e)}")
            if verbose:
                print(f"❌ EPO extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'archives_processed': 0,
                'xml_files_extracted': 0,
                'pdf_files_extracted': 0,
                'errors': 1
            }
    
    def run_full_pipeline(self, sources: List[str] = None, verbose: bool = True) -> Dict:
        """
        Run the full data pipeline for specified sources.
        
        Args:
            sources: List of sources to process ['epo', 'uspto'] (default: ['epo'])
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with overall pipeline statistics
        """
        if sources is None:
            sources = ['epo']
        
        start_time = time.time()
        results = {}
        
        if verbose:
            print("🚀 Starting Full Data Pipeline...")
            print(f"📂 Base directory: {self.base_dir}")
            print(f"🎯 Processing sources: {', '.join(sources)}")
        
        # Process EPO data
        if 'epo' in sources:
            if verbose:
                print("\n" + "="*50)
                print("📋 PROCESSING EPO PATENT DATA")
                print("="*50)
            
            try:
                epo_results = self.extract_epo_data(verbose=verbose)
                results['epo'] = epo_results
            except Exception as e:
                logger.error(f"EPO processing failed: {str(e)}")
                results['epo'] = {"success": False, "error": str(e)}
        
        # Process USPTO data (placeholder for future implementation)
        if 'uspto' in sources:
            if verbose:
                print("\n" + "="*50)
                print("🇺🇸 PROCESSING USPTO PATENT DATA")
                print("="*50)
                print("📝 USPTO processing not yet implemented")
            
            results['uspto'] = {"success": False, "message": "Not yet implemented"}
        
        # Calculate total time
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        
        results['pipeline_stats'] = {
            'total_time': total_time,
            'sources_processed': sources,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if verbose:
            print(f"\n🎉 Pipeline completed in {total_time} seconds!")
        
        return results


def run_epo_data_pipeline(base_dir: str = None, verbose: bool = True) -> Dict:
    """
    Convenience function to run EPO data pipeline.
    
    Args:
        base_dir: Base directory path (optional)
        verbose: Whether to print detailed progress information
        
    Returns:
        Dictionary with processing statistics
    """
    pipeline = DataPipeline(base_dir=base_dir)
    return pipeline.extract_epo_data(verbose=verbose)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Patent Data Pipeline')
    parser.add_argument('--sources', nargs='+', choices=['epo', 'uspto'], 
                       default=['epo'], help='Data sources to process')
    parser.add_argument('--base-dir', type=str, help='Base directory path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(base_dir=args.base_dir)
    results = pipeline.run_full_pipeline(
        sources=args.sources,
        verbose=args.verbose
    )
    
    print(f"\n📊 Final Results: {results}")


if __name__ == "__main__":
    main()