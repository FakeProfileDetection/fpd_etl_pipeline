#!/usr/bin/env python3
"""Integration tests for the complete pipeline"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pipeline import clean_data, extract_keypairs, extract_features
from scripts.utils.data_validator import validate_pipeline_output
from tests.test_utils import TestDataGenerator, create_test_config, setup_test_version_manager, cleanup_test_version_manager


class TestPipelineIntegration(unittest.TestCase):
    """Test the complete pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = create_test_config()
        self.config["ARTIFACTS_DIR"] = str(self.test_dir / "artifacts")
        
        self.version_id = "test_integration_001"
        self.data_gen = TestDataGenerator()
        
        # Setup test version manager
        self.vm = setup_test_version_manager(self.test_dir)
        self.vm.register_version(self.version_id, {"test": True})
        
    def tearDown(self):
        """Clean up test environment"""
        cleanup_test_version_manager()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_complete_pipeline_flow(self):
        """Test running all pipeline stages in sequence"""
        # Step 1: Create test input data
        input_dir = self.test_dir / "web_app_data"
        input_dir.mkdir(parents=True)
        
        # Create data for 3 users (32 hex chars each)
        user_ids = [
            "a" * 32,
            "b" * 32,
            "c" * 32
        ]
        
        for user_id in user_ids:
            self.data_gen.create_complete_user_data(input_dir, user_id)
            
        # Step 2: Run clean data stage
        clean_stage = clean_data.CleanDataStage(self.version_id, self.config, version_manager=self.vm)
        cleaned_dir = clean_stage.run(input_dir)
        
        self.assertTrue(cleaned_dir.exists())
        
        # Step 3: Run extract keypairs stage
        keypair_stage = extract_keypairs.ExtractKeypairsStage(self.version_id, self.config, version_manager=self.vm)
        keypair_dir = keypair_stage.run(cleaned_dir)
        
        self.assertTrue(keypair_dir.exists())
        self.assertTrue((keypair_dir / "keypairs.parquet").exists())
        
        # Step 4: Run extract features stage
        feature_stage = extract_features.ExtractFeaturesStage(self.version_id, self.config, version_manager=self.vm)
        feature_dir = feature_stage.run(keypair_dir)
        
        self.assertTrue(feature_dir.exists())
        
        # Step 5: Validate complete pipeline output
        validation = validate_pipeline_output(
            self.version_id,
            Path(self.config["ARTIFACTS_DIR"])
        )
        
        self.assertTrue(validation['valid'])
        
        # Check stage results
        self.assertIn('clean_data', validation['stages'])
        self.assertEqual(validation['stages']['clean_data']['total_complete'], 3)
        
        self.assertIn('extract_keypairs', validation['stages'])
        self.assertGreater(validation['stages']['extract_keypairs']['total_keypairs'], 0)
        
        self.assertIn('extract_features', validation['stages'])
        
    def test_pipeline_with_broken_data(self):
        """Test pipeline handling of incomplete user data"""
        input_dir = self.test_dir / "web_app_data"
        input_dir.mkdir(parents=True)
        
        # Create mix of complete and incomplete users (32 hex chars)
        complete_user = "d" * 32
        incomplete_user = "e" * 32
        
        self.data_gen.create_complete_user_data(input_dir, complete_user, include_all_files=True)
        self.data_gen.create_complete_user_data(input_dir, incomplete_user, include_all_files=False)
        
        # Run pipeline stages
        clean_stage = clean_data.CleanDataStage(self.version_id, self.config, version_manager=self.vm)
        cleaned_dir = clean_stage.run(input_dir)
        
        keypair_stage = extract_keypairs.ExtractKeypairsStage(self.version_id, self.config, version_manager=self.vm)
        keypair_dir = keypair_stage.run(cleaned_dir)
        
        feature_stage = extract_features.ExtractFeaturesStage(self.version_id, self.config, version_manager=self.vm)
        feature_dir = feature_stage.run(keypair_dir)
        
        # Validate
        validation = validate_pipeline_output(
            self.version_id,
            Path(self.config["ARTIFACTS_DIR"])
        )
        
        # Should have 1 complete and 1 broken user
        self.assertEqual(validation['stages']['clean_data']['total_complete'], 1)
        self.assertEqual(validation['stages']['clean_data']['total_broken'], 1)
        
        # Features should only be extracted for complete user
        self.assertEqual(validation['stages']['extract_keypairs']['unique_users'], 1)
        
    def test_pipeline_error_recovery(self):
        """Test pipeline behavior when a stage fails"""
        # Create minimal input that will cause issues
        input_dir = self.test_dir / "bad_data"
        input_dir.mkdir(parents=True)
        
        # Create a user with invalid CSV data (32 hex chars)
        user_id = "f" * 32
        user_dir = input_dir
        
        # Create required JSON files
        self.data_gen.create_user_info_json(user_dir / f"{user_id}_consent.json", 'consent')
        self.data_gen.create_user_info_json(user_dir / f"{user_id}_demographics.json", 'demographics')
        self.data_gen.create_user_info_json(user_dir / f"{user_id}_start_time.json", 'start_time')
        self.data_gen.create_user_info_json(user_dir / f"{user_id}_completion.json", 'completion')
        
        # Create an invalid CSV file
        bad_csv = user_dir / f"f_{user_id}_0.csv"
        bad_csv.write_text("Invalid,CSV,Format\nNo,Proper,Headers\n")
        
        # The pipeline should handle this gracefully
        clean_stage = clean_data.CleanDataStage(self.version_id, self.config, version_manager=self.vm)
        cleaned_dir = clean_stage.run(input_dir)
        
        # User should be in broken data due to invalid CSV
        validation = validate_pipeline_output(
            self.version_id,
            Path(self.config["ARTIFACTS_DIR"])
        )
        
        # The stage should complete but mark data as broken
        self.assertTrue(cleaned_dir.exists())


if __name__ == '__main__':
    unittest.main()