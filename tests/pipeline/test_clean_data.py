#!/usr/bin/env python3
"""Tests for clean_data stage"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pipeline.clean_data import CleanDataStage
from tests.test_utils import TestDataGenerator, TestValidator, create_test_config, setup_test_version_manager, cleanup_test_version_manager


class TestCleanDataStage(unittest.TestCase):
    """Test the clean data pipeline stage"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = create_test_config()
        self.config["ARTIFACTS_DIR"] = str(self.test_dir / "artifacts")
        
        self.version_id = "test_version_001"
        self.data_gen = TestDataGenerator()
        self.validator = TestValidator()
        
        # Setup test version manager
        self.vm = setup_test_version_manager(self.test_dir)
        self.vm.register_version(self.version_id, {"test": True})
        
    def tearDown(self):
        """Clean up test environment"""
        cleanup_test_version_manager()
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_clean_complete_user_data(self):
        """Test cleaning complete user data"""
        # Create input directory with complete user data
        input_dir = self.test_dir / "input_data"
        input_dir.mkdir(parents=True)
        
        # Create complete user data (32-char hex ID)
        user_id = "a1b2c3d4e5f6789012345678901234ef"
        self.data_gen.create_complete_user_data(input_dir, user_id, include_all_files=True)
        
        # Run clean data stage
        stage = CleanDataStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(input_dir)
        
        # Validate output structure
        validation = self.validator.validate_cleaned_data_structure(output_dir)
        
        self.assertTrue(validation['valid'], f"Validation errors: {validation['errors']}")
        self.assertEqual(validation['stats']['desktop']['complete_users'], 1)
        self.assertEqual(validation['stats']['desktop']['broken_users'], 0)
        
        # Check that user directory exists in raw_data
        user_dir = output_dir / "desktop" / "raw_data" / user_id
        self.assertTrue(user_dir.exists())
        
        # Check metadata file
        metadata_file = output_dir / "desktop" / "metadata" / "metadata.csv"
        self.assertTrue(metadata_file.exists())
        
    def test_clean_incomplete_user_data(self):
        """Test cleaning incomplete user data"""
        # Create input directory with incomplete user data
        input_dir = self.test_dir / "input_data"
        input_dir.mkdir(parents=True)
        
        # Create incomplete user data (missing completion file)
        user_id = "b2c3d4e5f67890123456789012345678"
        self.data_gen.create_complete_user_data(input_dir, user_id, include_all_files=False)
        
        # Run clean data stage
        stage = CleanDataStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(input_dir)
        
        # Validate output structure
        validation = self.validator.validate_cleaned_data_structure(output_dir)
        
        self.assertTrue(validation['valid'], f"Validation errors: {validation['errors']}")
        self.assertEqual(validation['stats']['desktop']['complete_users'], 0)
        self.assertEqual(validation['stats']['desktop']['broken_users'], 1)
        
        # Check that user directory exists in broken_data
        user_dir = output_dir / "desktop" / "broken_data" / user_id
        self.assertTrue(user_dir.exists())
        
    def test_clean_mixed_data(self):
        """Test cleaning mix of complete and incomplete users"""
        input_dir = self.test_dir / "input_data"
        input_dir.mkdir(parents=True)
        
        # Create multiple users with different completeness (32-char hex IDs)
        complete_users = ["c3d4e5f678901234567890123456789a", "d4e5f678901234567890123456789abc"]
        incomplete_users = ["e5f6789012345678901234567890abcd", "f67890123456789012345678901abcde"]
        
        for user_id in complete_users:
            self.data_gen.create_complete_user_data(input_dir, user_id, include_all_files=True)
            
        for user_id in incomplete_users:
            self.data_gen.create_complete_user_data(input_dir, user_id, include_all_files=False)
            
        # Run clean data stage
        stage = CleanDataStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(input_dir)
        
        # Validate
        validation = self.validator.validate_cleaned_data_structure(output_dir)
        
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['stats']['desktop']['complete_users'], 2)
        self.assertEqual(validation['stats']['desktop']['broken_users'], 2)
        
    def test_dry_run_mode(self):
        """Test dry run mode doesn't create files"""
        input_dir = self.test_dir / "input_data"
        input_dir.mkdir(parents=True)
        
        # Create test data (32-char hex ID)
        user_id = "1234567890abcdef1234567890abcdef"
        self.data_gen.create_complete_user_data(input_dir, user_id)
        
        # Run in dry run mode
        stage = CleanDataStage(self.version_id, self.config, dry_run=True, version_manager=self.vm)
        output_dir = stage.run(input_dir)
        
        # Check that output directory wasn't created
        self.assertFalse(output_dir.exists())
        
    def test_empty_input_directory(self):
        """Test handling of empty input directory"""
        input_dir = self.test_dir / "empty_input"
        input_dir.mkdir(parents=True)
        
        # Run clean data stage
        stage = CleanDataStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(input_dir)
        
        # Should still create directory structure
        validation = self.validator.validate_cleaned_data_structure(output_dir)
        self.assertTrue(validation['valid'], f"Validation errors: {validation['errors']}")
        self.assertEqual(validation['stats']['desktop']['complete_users'], 0)
        self.assertEqual(validation['stats']['desktop']['broken_users'], 0)
        
    def test_processing_summary(self):
        """Test that processing summary is generated correctly"""
        input_dir = self.test_dir / "input_data"
        input_dir.mkdir(parents=True)
        
        # Create test data (32-char hex ID)
        user_id = "fedcba0987654321fedcba0987654321"
        self.data_gen.create_complete_user_data(input_dir, user_id)
        
        # Run clean data stage
        stage = CleanDataStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(input_dir)
        
        # Check cleaning report
        report_file = output_dir.parent / "etl_metadata" / "cleaning" / "cleaning_report.json"
        self.assertTrue(report_file.exists())
        
        with open(report_file) as f:
            report = json.load(f)
            
        self.assertIn('summary', report)
        self.assertIn('total_users', report['summary'])
        self.assertIn('complete_users', report['summary'])
        self.assertIn('broken_users', report['summary'])
        self.assertEqual(report['summary']['total_users'], 1)
        self.assertEqual(report['summary']['complete_users'], 1)


if __name__ == '__main__':
    unittest.main()