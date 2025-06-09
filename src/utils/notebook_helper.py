def setup_notebook_environment():
    """Setup notebook environment with proper imports and config"""
    import sys, os
    from dotenv import load_dotenv
    
    # Add src to path
    src_path = os.path.abspath("../src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    # Load environment
    load_dotenv("../.env")
    
    return "Environment ready for experiments"