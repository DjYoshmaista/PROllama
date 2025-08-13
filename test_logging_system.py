#!/usr/bin/env python3
"""
Test script to validate the enhanced logging system for the folder processing pipeline
"""
import logging
import sys
import tempfile
import os

# Setup comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_logging_output.log', mode='w')
    ]
)

# Create root logger
logger = logging.getLogger(__name__)
LOG_PREFIX = "[TestLogging]"

def test_import_modules():
    """Test that all modules with enhanced logging can be imported"""
    print(f"{LOG_PREFIX} Testing module imports...")
    logger.info(f"{LOG_PREFIX} Starting module import tests")
    
    try:
        # Test file handler import
        from cli.handlers.file_handler import FileHandler, LOG_PREFIX as FH_PREFIX
        logger.info(f"{LOG_PREFIX} Successfully imported FileHandler with prefix: {FH_PREFIX}")
        
        # Test discovery import  
        from file_management.discovery import FileDiscovery, LOG_PREFIX as DISC_PREFIX
        logger.info(f"{LOG_PREFIX} Successfully imported FileDiscovery with prefix: {DISC_PREFIX}")
        
        # Test loader import
        from file_management.loaders import OptimizedBulkLoader, LOG_PREFIX as LOAD_PREFIX
        logger.info(f"{LOG_PREFIX} Successfully imported OptimizedBulkLoader with prefix: {LOAD_PREFIX}")
        
        # Test parser import
        from file_management.parsers import DocumentParser, LOG_PREFIX as PARSE_PREFIX
        logger.info(f"{LOG_PREFIX} Successfully imported DocumentParser with prefix: {PARSE_PREFIX}")
        
        # Test database import
        from database.operations import DatabaseOperations, LOG_PREFIX as DB_PREFIX
        logger.info(f"{LOG_PREFIX} Successfully imported DatabaseOperations with prefix: {DB_PREFIX}")
        
        print(f"{LOG_PREFIX} ✅ All modules imported successfully with logging prefixes")
        return True
        
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Module import failed: {e}")
        print(f"{LOG_PREFIX} ❌ Module import failed: {e}")
        return False

def test_logging_prefixes():
    """Test that logging prefixes are working correctly"""
    print(f"{LOG_PREFIX} Testing logging prefixes...")
    logger.info(f"{LOG_PREFIX} Starting prefix validation tests")
    
    try:
        # Test discovery logging
        from file_management.discovery import file_discovery
        logger.info(f"{LOG_PREFIX} Testing FileDiscovery logging...")
        
        # This should generate logs with [Discovery] prefix
        test_path = tempfile.mkdtemp()
        count = file_discovery.count_files(test_path)
        logger.info(f"{LOG_PREFIX} FileDiscovery.count_files() returned {count} for empty temp dir")
        
        # Test parser logging
        from file_management.parsers import document_parser
        logger.info(f"{LOG_PREFIX} Testing DocumentParser logging...")
        
        # This should generate logs with [Parser] prefix
        results = document_parser.parse_files_parallel([])
        logger.info(f"{LOG_PREFIX} DocumentParser.parse_files_parallel() returned {len(results)} results for empty list")
        
        # Test database logging
        from database.operations import db_ops
        logger.info(f"{LOG_PREFIX} Testing DatabaseOperations logging...")
        
        # This should generate logs with [Database] prefix
        count = db_ops.get_document_count()
        logger.info(f"{LOG_PREFIX} DatabaseOperations.get_document_count() returned {count}")
        
        # Cleanup
        os.rmdir(test_path)
        
        print(f"{LOG_PREFIX} ✅ All logging prefixes tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Prefix testing failed: {e}")
        print(f"{LOG_PREFIX} ❌ Prefix testing failed: {e}")
        return False

def test_log_output():
    """Test that log output is being written to file"""
    print(f"{LOG_PREFIX} Testing log output...")
    logger.info(f"{LOG_PREFIX} Starting log output validation")
    
    try:
        # Force flush logs
        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Check if log file was created and has content
        if os.path.exists('test_logging_output.log'):
            with open('test_logging_output.log', 'r') as f:
                content = f.read()
                line_count = len(content.splitlines())
                
            logger.info(f"{LOG_PREFIX} Log file contains {line_count} lines")
            print(f"{LOG_PREFIX} ✅ Log file created with {line_count} lines")
            
            # Check for prefixes in log content
            prefixes_found = {
                '[TestLogging]': '[TestLogging]' in content,
                '[Discovery]': '[Discovery]' in content,
                '[Parser]': '[Parser]' in content,
                '[Database]': '[Database]' in content
            }
            
            print(f"{LOG_PREFIX} Prefix validation:")
            for prefix, found in prefixes_found.items():
                status = "✅" if found else "❌"
                print(f"{LOG_PREFIX}   {prefix}: {status}")
                logger.info(f"{LOG_PREFIX} Prefix {prefix}: {'found' if found else 'NOT found'}")
            
            return all(prefixes_found.values())
        else:
            print(f"{LOG_PREFIX} ❌ Log file was not created")
            return False
            
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Log output testing failed: {e}")
        print(f"{LOG_PREFIX} ❌ Log output testing failed: {e}")
        return False

def main():
    """Main test function"""
    print(f"{LOG_PREFIX} Starting enhanced logging system validation")
    print(f"{LOG_PREFIX} {'='*50}")
    
    logger.info(f"{LOG_PREFIX} Enhanced logging system test started")
    
    # Test 1: Module imports
    test1_passed = test_import_modules()
    print()
    
    # Test 2: Logging prefixes
    test2_passed = test_logging_prefixes() 
    print()
    
    # Test 3: Log output
    test3_passed = test_log_output()
    print()
    
    # Summary
    all_passed = test1_passed and test2_passed and test3_passed
    
    print(f"{LOG_PREFIX} {'='*50}")
    print(f"{LOG_PREFIX} TEST SUMMARY:")
    print(f"{LOG_PREFIX} Module imports: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"{LOG_PREFIX} Logging prefixes: {'✅ PASSED' if test2_passed else '❌ FAILED'}")  
    print(f"{LOG_PREFIX} Log output: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"{LOG_PREFIX} Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    logger.info(f"{LOG_PREFIX} Enhanced logging test completed - {'SUCCESS' if all_passed else 'FAILURE'}")
    
    if all_passed:
        print(f"\n{LOG_PREFIX} 🎉 The enhanced logging system is working correctly!")
        print(f"{LOG_PREFIX} You should now see detailed logs with prefixes when running menu option 4.")
        print(f"{LOG_PREFIX} Check 'test_logging_output.log' for the complete log output.")
    else:
        print(f"\n{LOG_PREFIX} ⚠️  Some logging tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)