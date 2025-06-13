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

# Bootstrap function that can be called directly
def bootstrap_and_setup():
    """Bootstrap the environment and return the setup function"""
    import sys, os
    
    # Add src to path first
    src_path = os.path.abspath("../src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    # Now call the main setup
    return setup_notebook_environment()