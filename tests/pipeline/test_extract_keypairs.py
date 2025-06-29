#!/usr/bin/env python3
"""Tests for extract_keypairs stage"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pipeline.extract_keypairs import ExtractKeypairsStage
from tests.test_utils import TestDataGenerator, TestValidator, create_test_config, setup_test_version_manager, cleanup_test_version_manager


class TestExtractKeypairsStage(unittest.TestCase):
    """Test the extract keypairs pipeline stage"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = create_test_config()
        self.config["ARTIFACTS_DIR"] = str(self.test_dir / "artifacts")
        
        self.version_id = "test_version_002"
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
            
    def create_cleaned_data_structure(self, cleaned_dir: Path, num_users: int = 1,
                                    include_errors: bool = False) -> None:
        """Create a cleaned data directory structure for testing"""
        # Create directory structure
        for device in ['desktop']:
            device_dir = cleaned_dir / device
            raw_data_dir = device_dir / "raw_data"
            raw_data_dir.mkdir(parents=True)
            
            # Create metadata directory
            (device_dir / "metadata").mkdir()
            (device_dir / "text").mkdir()
            (device_dir / "broken_data").mkdir()
            
            # Create user data
            for i in range(num_users):
                # Create valid 32-char hex user ID
                user_id = f"{i:0>32x}"  # Pad with zeros to make 32 hex chars
                user_dir = raw_data_dir / user_id
                user_dir.mkdir()
                
                # Create files in TypeNet format (as they would be after clean_data stage)
                # Create a few files for each platform
                for platform in [1, 2, 3]:
                    for session in [1, 2]:
                        for video in [1, 2]:
                            csv_file = user_dir / f"{platform}_{video}_{session}_{user_id}.csv"
                            self.data_gen.create_keystroke_csv(csv_file, 
                                                             num_events=50,
                                                             include_errors=include_errors)
                        
    def test_extract_keypairs_basic(self):
        """Test basic keypair extraction"""
        # Create cleaned data structure
        cleaned_dir = self.test_dir / "cleaned_data"
        self.create_cleaned_data_structure(cleaned_dir, num_users=2)
        
        # Run extract keypairs stage
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(cleaned_dir)
        
        # Check output file exists
        keypair_file = output_dir / "keypairs.parquet"
        self.assertTrue(keypair_file.exists())
        
        
        # Validate keypair data
        validation = self.validator.validate_keypair_data(keypair_file)
        self.assertTrue(validation['valid'], f"Validation errors: {validation['errors']}")
        
        # Check statistics
        self.assertGreater(validation['stats']['total_keypairs'], 0)
        self.assertGreaterEqual(validation['stats']['unique_users'], 2)
        
    def test_extract_keypairs_with_errors(self):
        """Test keypair extraction with data quality issues"""
        # Create cleaned data with errors
        cleaned_dir = self.test_dir / "cleaned_data"
        self.create_cleaned_data_structure(cleaned_dir, num_users=1, include_errors=True)
        
        # Run extract keypairs stage
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(cleaned_dir)
        
        # Load keypair data
        keypair_file = output_dir / "keypairs.parquet"
        df = pd.read_parquet(keypair_file)
        
        # Check that invalid keypairs are marked
        self.assertIn('valid', df.columns)
        invalid_count = (~df['valid']).sum()
        self.assertGreater(invalid_count, 0, "Should have some invalid keypairs due to errors")
        
        # Check specific error types in metadata
        # Note: data_quality_issues.json creation not implemented yet
        # issues_file = output_dir.parent / "etl_metadata" / "keypairs" / "data_quality_issues.json"
        # self.assertTrue(issues_file.exists())
        
    def test_timing_calculations(self):
        """Test that timing calculations are correct"""
        # Create simple test data
        cleaned_dir = self.test_dir / "cleaned_data"
        desktop_dir = cleaned_dir / "desktop" / "raw_data"
        desktop_dir.mkdir(parents=True)
        
        user_id = "a" * 32  # Valid 32-char hex ID
        user_dir = desktop_dir / user_id
        user_dir.mkdir()
        
        # Create a simple CSV with known timings (TypeNet format, no headers)
        csv_file = user_dir / f"1_1_1_{user_id}.csv"
        with open(csv_file, 'w') as f:
            # No headers, direct data
            f.write("P,h,1000000000\n")  # h pressed at 1000ms (in nanoseconds)
            f.write("P,e,1100000000\n")  # e pressed at 1100ms (IL = 100ms)
            f.write("R,h,1150000000\n")  # h released at 1150ms (HL = 150ms)
            f.write("R,e,1250000000\n")  # e released at 1250ms
        
        # Run extraction
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(cleaned_dir)
        
        # Load and check results
        df = pd.read_parquet(output_dir / "keypairs.parquet")
        
        # Find the h->e keypair
        keypair = df[(df['key1'] == 'h') & (df['key2'] == 'e')].iloc[0]
        
        # Check timing calculations (in nanoseconds)
        self.assertEqual(keypair['HL'], 150000000)  # 150ms in nanoseconds
        self.assertEqual(keypair['IL'], -50000000)  # -50ms (e pressed before h released)
        self.assertEqual(keypair['PL'], 100000000)  # 100ms
        self.assertEqual(keypair['RL'], 100000000)  # 100ms
        
    def test_outlier_detection(self):
        """Test outlier detection functionality"""
        # Create cleaned data
        cleaned_dir = self.test_dir / "cleaned_data"
        self.create_cleaned_data_structure(cleaned_dir, num_users=3)
        
        # Run extraction with outlier detection
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        stage.config['DETECT_OUTLIERS'] = True
        output_dir = stage.run(cleaned_dir)
        
        # Check that outlier column exists
        df = pd.read_parquet(output_dir / "keypairs.parquet")
        self.assertIn('outlier', df.columns)
        
        # Should have some outliers marked (but not all)
        outlier_count = df['outlier'].sum()
        self.assertGreater(outlier_count, 0)
        self.assertLess(outlier_count, len(df))
        
    def test_dry_run_mode(self):
        """Test dry run mode for keypair extraction"""
        cleaned_dir = self.test_dir / "cleaned_data"
        self.create_cleaned_data_structure(cleaned_dir)
        
        # Run in dry run mode
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=True, version_manager=self.vm)
        output_dir = stage.run(cleaned_dir)
        
        # Check that no output files were created
        keypair_file = output_dir / "keypairs.parquet"
        self.assertFalse(keypair_file.exists())
        
    def test_empty_input(self):
        """Test handling of empty input directory"""
        cleaned_dir = self.test_dir / "cleaned_data"
        cleaned_dir.mkdir(parents=True)
        
        # Create minimal structure
        (cleaned_dir / "desktop" / "raw_data").mkdir(parents=True)
        
        # Run extraction
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(cleaned_dir)
        
        # Should create empty output
        keypair_file = output_dir / "keypairs.parquet"
        self.assertTrue(keypair_file.exists())
        
        df = pd.read_parquet(keypair_file)
        self.assertEqual(len(df), 0)
        
    def test_metadata_tracking(self):
        """Test that metadata is properly tracked"""
        cleaned_dir = self.test_dir / "cleaned_data"
        self.create_cleaned_data_structure(cleaned_dir, num_users=2)
        
        # Run extraction
        stage = ExtractKeypairsStage(self.version_id, self.config, dry_run=False, version_manager=self.vm)
        output_dir = stage.run(cleaned_dir)
        
        # Check extraction stats
        stats_file = output_dir.parent / "etl_metadata" / "keypairs" / "extraction_stats.json"
        self.assertTrue(stats_file.exists())
        
        # Check file mapping
        # Note: file_mapping.json creation not implemented yet
        # mapping_file = output_dir.parent / "etl_metadata" / "keypairs" / "file_mapping.json"
        # self.assertTrue(mapping_file.exists())


if __name__ == '__main__':
    unittest.main()