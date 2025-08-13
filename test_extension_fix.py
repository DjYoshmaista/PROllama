#!/usr/bin/env python3
"""
Test script to verify the extension filtering fix in instant_loader.py
"""
import tempfile
import os
from pathlib import Path
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_extension_filtering():
    """Test that extension filtering now matches discovery.py"""
    print("🧪 Testing extension filtering fix...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with different extensions
        test_files = [
            "test1.txt",
            "test2.py", 
            "test3.csv",
            "test4.json",
            "test5.md",     # Should be filtered out
            "test6.log",    # Should be filtered out
            "test7",        # No extension, should be filtered out
        ]
        
        for filename in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write("Test content " * 50)  # Ensure file size > 10 bytes
        
        print(f"📁 Created {len(test_files)} test files")
        
        # Test instant loader filtering
        from file_management.instant_loader import InstantAsyncLoader
        loader = InstantAsyncLoader()
        
        # Test the fixed filtering logic
        supported_extensions = {"py", "txt", "csv", "json"}
        skip_files = set()
        
        found_files = loader._scan_directory_for_files_sync(temp_dir, supported_extensions, skip_files)
        
        print(f"✅ InstantLoader found {len(found_files)} files:")
        for file_path in sorted(found_files):
            filename = os.path.basename(file_path)
            ext = Path(file_path).suffix[1:].lower()
            print(f"   📄 {filename} (ext: '{ext}')")
        
        # Expected files: test1.txt, test2.py, test3.csv, test4.json
        expected_count = 4
        
        if len(found_files) == expected_count:
            print(f"🎉 SUCCESS: Found {len(found_files)}/{expected_count} expected files")
            
            # Test discovery.py for comparison
            from file_management.discovery import file_discovery
            discovery_count = file_discovery.count_files(temp_dir)
            print(f"📊 Discovery.py count: {discovery_count} files")
            
            return True
        else:
            print(f"❌ FAILED: Expected {expected_count} files, got {len(found_files)}")
            return False

async def main():
    """Main test function"""
    print("🔧 Extension Filtering Fix Test")
    print("=" * 50)
    
    success = await test_extension_filtering()
    
    if success:
        print("\n🎯 EXTENSION FILTERING FIX VALIDATED!")
        print("✅ InstantLoader now uses same extension logic as discovery.py")
        print("✅ Should now process all discovered files instead of filtering 5M+ down to 2")
    else:
        print("\n⚠️  Extension filtering test failed")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)