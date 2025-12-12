#!/usr/bin/env python3
"""
Simple test script to check basic imports and functionality
"""

def test_imports():
    """Test basic module imports"""
    try:
        import sys
        import os
        from pathlib import Path
        
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))

        print("Testing basic imports...")

        # Test utils imports
        from utils.config import Config
        print("✓ utils.config imported successfully")

        from utils.logger import setup_logger
        print("✓ utils.logger imported successfully")

        # Test config loading
        config = Config()
        print("✓ Config loaded successfully")
        print(f"  Dataset paths: {list(config.dataset_paths.keys())}")

        # Test logger setup
        logger = setup_logger('test')
        logger.info("Test log message")
        print("✓ Logger setup successful")

        print("\nAll basic imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
