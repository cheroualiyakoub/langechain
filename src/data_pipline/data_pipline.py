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
xml_loader_dir = src_dir / "data_pipline" / "EU_XML_data_loader"
sys.path.append(str(eu_loader_dir))
sys.path.append(str(xml_loader_dir))

from load_raw_data_xml_pdf_EPO import extract_epo_archives
from xml_loader_EPO import process_xml_files_list

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
        self.epo_processed_dir = self.processed_dir / "EPO"
        
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
            self.epo_processed_dir,
            self.uspto_raw_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def get_epo_xml_files(self) -> List[str]:
        """
        Get all XML file paths from the EPO raw data directory.
        
        Returns:
            List of XML file paths
        """
        xml_files = []
        
        if not self.epo_raw_dir.exists():
            logger.warning(f"EPO raw directory does not exist: {self.epo_raw_dir}")
            return xml_files
        
        # Recursively find all XML files, excluding TOC.xml files
        for xml_file in self.epo_raw_dir.rglob("*.xml"):
            if xml_file.is_file() and not xml_file.name.lower().startswith('toc.'):
                xml_files.append(str(xml_file))
        
        logger.info(f"Found {len(xml_files)} XML files in EPO raw directory")
        return xml_files
    
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

    def process_epo_xml_to_json(self, debug: bool = False, verbose: bool = True) -> Dict:   
        """
        Process all EPO XML files to JSON format.
        
        Args:
            debug: Whether to enable debug output during XML processing
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting EPO XML to JSON processing...")
        
        # Ensure directories exist
        self.ensure_directories()
        
        if verbose:
            print(f"\n🔄 Starting EPO XML to JSON Processing")
            print(f"📂 Raw data directory: {self.epo_raw_dir}")
            print(f"📁 Processed directory: {self.epo_processed_dir}")
        
        # Get all XML files
        xml_files = self.get_epo_xml_files()
        
        if not xml_files:
            message = "No XML files found to process"
            logger.warning(message)
            if verbose:
                print(f"⚠️ {message}")
            return {
                'success': False,
                'message': message,
                'xml_files_found': 0,
                'processed_count': 0,
                'error_count': 0
            }
        
        if verbose:
            print(f"📋 Found {len(xml_files)} XML files to process")
        
        # Process XML files using the imported function
        try:
            start_time = time.time()
            
            # Call the processing function from xml_loader_EPO
            process_xml_files_list(xml_files, debug=debug)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Count resulting JSON files to verify success
            json_files = list(self.epo_processed_dir.rglob("*.json"))
            
            result = {
                'success': True,
                'xml_files_found': len(xml_files),
                'json_files_created': len(json_files),
                'processing_time': processing_time,
                'debug_enabled': debug
            }
            
            if verbose:
                print(f"\n✅ EPO XML processing completed!")
                print(f"⏱️  Processing time: {processing_time} seconds")
                print(f"📊 JSON files created: {len(json_files)}")
            
            logger.info(f"EPO XML processing completed: {len(json_files)} JSON files created")
            return result
            
        except Exception as e:
            logger.error(f"EPO XML processing failed: {str(e)}")
        if verbose:
            print(f"❌ EPO XML processing failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'xml_files_found': len(xml_files),
            'processed_count': 0,
            'error_count': 1
        }

def run_full_pipeline(self, sources: List[str] = None, process_xml: bool = True, 
                        debug_xml: bool = False, verbose: bool = True) -> Dict:
    """
    Run the full data pipeline for specified sources.
    
    Args:
        sources: List of sources to process ['epo', 'uspto'] (default: ['epo'])
        process_xml: Whether to process XML files to JSON after extraction
        debug_xml: Whether to enable debug output during XML processing
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
        print(f"📋 XML processing: {'Enabled' if process_xml else 'Disabled'}")
    
    # Process EPO data
    if 'epo' in sources:
        if verbose:
            print("\n" + "="*50)
            print("📋 PROCESSING EPO PATENT DATA")
            print("="*50)
        
        try:
            # Step 1: Extract archives
            epo_extraction_results = self.extract_epo_data(verbose=verbose)
            results['epo_extraction'] = epo_extraction_results
            
            # Step 2: Process XML to JSON (if enabled and extraction was successful)
            if process_xml and epo_extraction_results.get('success', True):
                epo_processing_results = self.process_epo_xml_to_json(
                    debug=debug_xml, 
                    verbose=verbose
                )
                results['epo_xml_processing'] = epo_processing_results
            elif process_xml:
                if verbose:
                    print("⏩ Skipping XML processing due to extraction failure")
                results['epo_xml_processing'] = {
                    'success': False, 
                    'message': 'Skipped due to extraction failure'
                }
            
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
        'xml_processing_enabled': process_xml,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if verbose:
        print(f"\n🎉 Full Pipeline completed in {total_time} seconds!")
        
        # Summary
        print(f"\n📊 Pipeline Summary:")
        if 'epo_extraction' in results:
            extract_stats = results['epo_extraction']
            print(f"  📦 EPO Extraction: {extract_stats.get('xml_files_extracted', 0)} XML + {extract_stats.get('pdf_files_extracted', 0)} PDF files")
        
        if 'epo_xml_processing' in results:
            process_stats = results['epo_xml_processing']
            if process_stats.get('success'):
                print(f"  🔄 EPO Processing: {process_stats.get('json_files_created', 0)} JSON files created")
            else:
                print(f"  ❌ EPO Processing: Failed")
    
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


def run_epo_xml_processing(base_dir: str = None, debug: bool = False, verbose: bool = True) -> Dict:
    """
    Convenience function to run only EPO XML processing.
    
    Args:
        base_dir: Base directory path (optional)
        debug: Whether to enable debug output during XML processing
        verbose: Whether to print detailed progress information
        
    Returns:
        Dictionary with processing statistics
    """
    pipeline = DataPipeline(base_dir=base_dir)
    return pipeline.process_epo_xml_to_json(debug=debug, verbose=verbose)

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Patent Data Pipeline')
    parser.add_argument('--sources', nargs='+', choices=['epo', 'uspto'], 
                       default=['epo'], help='Data sources to process')
    parser.add_argument('--base-dir', type=str, help='Base directory path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-xml', action='store_true', help='Skip XML processing step')
    parser.add_argument('--debug-xml', action='store_true', help='Enable debug output for XML processing')
    parser.add_argument('--xml-only', action='store_true', help='Run only XML processing (skip archive extraction)')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(base_dir=args.base_dir)
    
    if args.xml_only:
        # Run only XML processing
        results = pipeline.process_epo_xml_to_json(
            debug=args.debug_xml,
            verbose=args.verbose
        )
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            sources=args.sources,
            process_xml=not args.skip_xml,
            debug_xml=args.debug_xml,
            verbose=args.verbose
        )
    
    print(f"\n📊 Final Results: {results}")

if __name__ == "__main__":
    main()