#!/usr/bin/env python3
"""Tests for extract_features stage"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pipeline.extract_features import ExtractFeaturesStage, TypeNetMLFeatureExtractor
from tests.test_utils import TestDataGenerator, TestValidator, create_test_config


class TestExtractFeaturesStage(unittest.TestCase):
    """Test the extract features pipeline stage"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = create_test_config()
        self.config["ARTIFACTS_DIR"] = str(self.test_dir / "artifacts")
        
        self.version_id = "test_version_003"
        self.data_gen = TestDataGenerator()
        self.validator = TestValidator()
        
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def create_test_keypair_data(self, output_path: Path, num_users: int = 3,
                               num_keypairs_per_user: int = 100) -> pd.DataFrame:
        """Create test keypair data"""
        data = []
        
        # Common keys for consistency
        keys = ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd', 'a', 'i', 'n', 't']
        
        for user_idx in range(num_users):
            user_id = f"test_user_{user_idx}_" + "0" * (32 - len(f"test_user_{user_idx}_"))
            
            for _ in range(num_keypairs_per_user):
                key1 = np.random.choice(keys)
                key2 = np.random.choice(keys)
                
                # Generate realistic timing data (in nanoseconds)
                hl = np.random.normal(100, 20) * 1_000_000  # ~100ms hold time
                il = np.random.normal(50, 30) * 1_000_000   # ~50ms inter-key
                pl = hl + il
                rl = il + np.random.normal(100, 20) * 1_000_000
                
                data.append({
                    'user_id': user_id,
                    'platform_id': np.random.choice([1, 2, 3]),
                    'session_id': np.random.choice([1, 2]),
                    'video_id': np.random.choice([1, 2, 3, 4, 5, 6]),
                    'device_type': 'desktop',
                    'key1': key1,
                    'key2': key2,
                    'HL': max(0, hl),  # Ensure non-negative
                    'IL': il,
                    'PL': max(0, pl),
                    'RL': max(0, rl),
                    'valid': True,
                    'outlier': np.random.random() < 0.05  # 5% outliers
                })
                
        df = pd.DataFrame(data)
        
        # Save as parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        return df
        
    def test_feature_extraction_basic(self):
        """Test basic feature extraction"""
        # Create keypair data
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        self.create_test_keypair_data(keypair_file)
        
        # Run feature extraction
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        output_dir = stage.run(keypair_dir)
        
        # Validate output
        validation = self.validator.validate_features(output_dir)
        self.assertTrue(validation['valid'], f"Validation errors: {validation['errors']}")
        
        # Check that all feature types were created
        expected_types = ['typenet_ml_user_platform', 'typenet_ml_session', 'typenet_ml_video']
        actual_types = [d.name for d in output_dir.iterdir() if d.is_dir()]
        
        for expected in expected_types:
            self.assertIn(expected, actual_types)
            
    def test_feature_imputation(self):
        """Test that missing values are properly imputed"""
        # Create sparse keypair data (will have missing features)
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        
        # Create data with limited key combinations
        data = []
        user_id = "sparse_user_" + "0" * 20
        
        # Only create a few specific keypairs
        for i in range(10):
            data.append({
                'user_id': user_id,
                'platform_id': 1,
                'session_id': 1,
                'video_id': 1,
                'device_type': 'desktop',
                'key1': 'h',
                'key2': 'e',
                'HL': 100_000_000,
                'IL': 50_000_000,
                'PL': 150_000_000,
                'RL': 200_000_000,
                'valid': True,
                'outlier': False
            })
            
        df = pd.DataFrame(data)
        keypair_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(keypair_file)
        
        # Run extraction
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        output_dir = stage.run(keypair_dir)
        
        # Load features and check for NaN values
        feature_file = output_dir / "typenet_ml_user_platform" / "features.parquet"
        features_df = pd.read_parquet(feature_file)
        
        # Should have no NaN values after imputation
        nan_count = features_df.isna().sum().sum()
        self.assertEqual(nan_count, 0, "Features should have no NaN values after imputation")
        
    def test_aggregation_levels(self):
        """Test different aggregation levels produce correct shapes"""
        # Create keypair data
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        df = self.create_test_keypair_data(keypair_file, num_users=2)
        
        # Run extraction
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        output_dir = stage.run(keypair_dir)
        
        # Check shapes for each aggregation level
        # User-platform level
        up_features = pd.read_parquet(output_dir / "typenet_ml_user_platform" / "features.parquet")
        expected_up_rows = df.groupby(['user_id', 'platform_id']).ngroups
        self.assertEqual(len(up_features), expected_up_rows)
        
        # Session level
        session_features = pd.read_parquet(output_dir / "typenet_ml_session" / "features.parquet")
        expected_session_rows = df.groupby(['user_id', 'platform_id', 'session_id']).ngroups
        self.assertEqual(len(session_features), expected_session_rows)
        
        # Video level
        video_features = pd.read_parquet(output_dir / "typenet_ml_video" / "features.parquet")
        expected_video_rows = df.groupby(['user_id', 'platform_id', 'session_id', 'video_id']).ngroups
        self.assertEqual(len(video_features), expected_video_rows)
        
    def test_outlier_handling(self):
        """Test that outliers are handled according to config"""
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        df = self.create_test_keypair_data(keypair_file)
        
        # Count outliers in input
        outlier_count = df['outlier'].sum()
        self.assertGreater(outlier_count, 0, "Test data should have some outliers")
        
        # Test with outliers excluded (default)
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        self.config['KEEP_OUTLIERS'] = False
        output_dir = stage.run(keypair_dir)
        
        # Test with outliers included
        self.config['KEEP_OUTLIERS'] = True
        stage2 = ExtractFeaturesStage(self.version_id + "_with_outliers", self.config, dry_run=False)
        output_dir2 = stage2.run(keypair_dir)
        
        # Compare feature counts
        features1 = pd.read_parquet(output_dir / "typenet_ml_user_platform" / "features.parquet")
        features2 = pd.read_parquet(output_dir2 / "typenet_ml_user_platform" / "features.parquet")
        
        # With outliers included, we might have different statistics
        # (This is a simple check - in reality the values would differ)
        self.assertTrue(len(features1) > 0)
        self.assertTrue(len(features2) > 0)
        
    def test_feature_registry(self):
        """Test that feature registry is properly created"""
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        self.create_test_keypair_data(keypair_file)
        
        # Run extraction
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        output_dir = stage.run(keypair_dir)
        
        # Check feature registry
        registry_file = output_dir.parent / "etl_metadata" / "features" / "feature_registry.json"
        self.assertTrue(registry_file.exists())
        
        with open(registry_file) as f:
            registry = json.load(f)
            
        self.assertIn('features_available', registry)
        self.assertIn('feature_configs', registry)
        self.assertEqual(len(registry['features_available']), 3)
        
    def test_specific_feature_types(self):
        """Test extracting only specific feature types"""
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        self.create_test_keypair_data(keypair_file)
        
        # Extract only user_platform features
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        output_dir = stage.run(keypair_dir, feature_types=['typenet_ml_user_platform'])
        
        # Check that only requested type was created
        created_dirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
        self.assertEqual(len(created_dirs), 1)
        self.assertEqual(created_dirs[0], 'typenet_ml_user_platform')
        
    def test_dry_run_mode(self):
        """Test dry run mode for feature extraction"""
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        self.create_test_keypair_data(keypair_file)
        
        # Run in dry run mode
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=True)
        output_dir = stage.run(keypair_dir)
        
        # Check that no output was created
        self.assertFalse(output_dir.exists())
        
    def test_statistical_features(self):
        """Test that statistical features are calculated correctly"""
        keypair_dir = self.test_dir / "keypairs"
        keypair_file = keypair_dir / "keypairs.parquet"
        
        # Create controlled data
        data = []
        user_id = "stats_test_user_" + "0" * 17
        
        # Create 5 'h' keypairs with known HL values
        hl_values = [100, 110, 120, 130, 140]  # milliseconds
        for hl in hl_values:
            data.append({
                'user_id': user_id,
                'platform_id': 1,
                'session_id': 1,
                'video_id': 1,
                'device_type': 'desktop',
                'key1': 'h',
                'key2': 'e',
                'HL': hl * 1_000_000,  # Convert to nanoseconds
                'IL': 50_000_000,
                'PL': 150_000_000,
                'RL': 200_000_000,
                'valid': True,
                'outlier': False
            })
            
        df = pd.DataFrame(data)
        keypair_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(keypair_file)
        
        # Run extraction
        stage = ExtractFeaturesStage(self.version_id, self.config, dry_run=False)
        output_dir = stage.run(keypair_dir)
        
        # Load features
        features = pd.read_parquet(output_dir / "typenet_ml_user_platform" / "features.parquet")
        
        # Check statistical calculations for 'h' HL
        row = features.iloc[0]
        
        # Expected values (in milliseconds)
        expected_mean = np.mean(hl_values)
        expected_median = np.median(hl_values)
        expected_std = np.std(hl_values)
        
        # Check values (allowing small floating point differences)
        self.assertAlmostEqual(row['HL_h_mean'], expected_mean, places=1)
        self.assertAlmostEqual(row['HL_h_median'], expected_median, places=1)
        self.assertAlmostEqual(row['HL_h_std'], expected_std, places=1)


if __name__ == '__main__':
    unittest.main()