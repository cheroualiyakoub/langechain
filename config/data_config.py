from pathlib import Path

# Get project root directory (go up one level from config directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Define data directories
RAW_JSON_DIR = str(PROJECT_ROOT / "data" / "parsed" / "EPO")
RAW_DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
parsed_DATA_DIR = str(PROJECT_ROOT / "data" / "parsed")
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "data" / "processed")  # Keep both for compatibility
ARCHIVE_DIR = str(PROJECT_ROOT / "data" / "archive")