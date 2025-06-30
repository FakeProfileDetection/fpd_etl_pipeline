"""
Integration tests for pipeline data flow
Ensures data moves correctly through all stages
"""

import pytest
from pathlib import Path
import shutil
import tempfile
from scripts.utils.test_data_generator import FakeDataGenerator
from scripts.utils.version_manager import VersionManager
from scripts.pipeline import download_data, clean_data, extract_keypairs, extract_features


class TestPipelineDataFlow:
    """Test data flow through pipeline stages"""
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_version_id(self):
        """Create test version ID"""
        return f"test_integration_{id(self)}"
    
    @pytest.fixture
    def test_config(self, temp_artifacts_dir):
        """Create test configuration"""
        return {
            "ARTIFACTS_DIR": str(temp_artifacts_dir),
            "DEVICE_TYPES": ["desktop"],
            "UPLOAD_ARTIFACTS": False,
            "INCLUDE_PII": False,
            "GENERATE_REPORTS": False
        }
    
    def test_download_creates_directory_structure(self, test_version_id, test_config):
        """Test that download stage creates proper directory structure"""
        # Create test data
        test_data_dir = Path("test_data/raw_data")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        generator = FakeDataGenerator()
        # Create one complete user
        generator.generate_complete_user(test_data_dir, "a" * 32)
        
        try:
            # Run download stage
            output_dir = download_data.run(
                version_id=test_version_id,
                config=test_config,
                dry_run=False,
                local_only=True
            )
            
            # Verify directory structure
            assert output_dir.exists()
            assert output_dir.name == "web_app_data"
            assert output_dir.parent.name == "raw_data"
            
            # Verify files were copied
            files = list(output_dir.glob("*"))
            assert len(files) > 0
            
            # Check for expected file types
            csv_files = list(output_dir.glob("*.csv"))
            json_files = list(output_dir.glob("*.json"))
            assert len(csv_files) >= 18  # At least one complete user
            assert len(json_files) >= 3   # consent, demographics, start_time
            
        finally:
            # Cleanup
            shutil.rmtree(test_data_dir)
    
    def test_clean_organizes_users_correctly(self, test_version_id, test_config):
        """Test that clean stage organizes users into correct directories"""
        # Setup test data
        artifacts_dir = Path(test_config["ARTIFACTS_DIR"]) / test_version_id
        raw_data_dir = artifacts_dir / "raw_data" / "web_app_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        generator = FakeDataGenerator()
        
        # Create complete user with predictable ID
        complete_user_id = generator.generate_complete_user(raw_data_dir, "complete")
        
        # Create incomplete user (missing files) 
        incomplete_user_id = generator.generate_user_files(raw_data_dir, "incomplete", num_files=5)
        
        # Register version and simulate download stage completion
        vm = VersionManager()
        vm.register_version(test_version_id, {"test": True})
        vm.update_stage_info(test_version_id, "download_data", {
            "output_path": str(raw_data_dir),
            "completed": True
        })
        
        # Run clean stage
        output_dir = clean_data.run(
            version_id=test_version_id,
            config=test_config,
            dry_run=False,
            local_only=True
        )
        
        # Verify directory structure
        assert output_dir.exists()
        assert output_dir.name == "cleaned_data"
        
        desktop_dir = output_dir / "desktop"
        assert desktop_dir.exists()
        
        raw_data_dir = desktop_dir / "raw_data"
        broken_data_dir = desktop_dir / "broken_data"
        metadata_dir = desktop_dir / "metadata"
        
        assert raw_data_dir.exists()
        assert broken_data_dir.exists()
        assert metadata_dir.exists()
        
        # Check complete user in raw_data
        complete_user_dir = raw_data_dir / complete_user_id
        assert complete_user_dir.exists()
        assert len(list(complete_user_dir.glob("*.csv"))) == 18
        
        # Check incomplete user in broken_data
        incomplete_user_dir = broken_data_dir / incomplete_user_id
        assert incomplete_user_dir.exists()
        assert len(list(incomplete_user_dir.glob("*.csv"))) < 18
        
        # Check metadata files
        assert (metadata_dir / "complete_users.txt").exists()
        assert (metadata_dir / "broken_users.txt").exists()
        assert (metadata_dir / "metadata.csv").exists()
    
    def test_keypairs_processes_clean_data(self, test_version_id, test_config):
        """Test that keypairs stage processes cleaned data correctly"""
        # Setup cleaned data
        artifacts_dir = Path(test_config["ARTIFACTS_DIR"]) / test_version_id
        cleaned_dir = artifacts_dir / "cleaned_data" / "desktop" / "raw_data"
        user_dir = cleaned_dir / ("test" + "a" * 28)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simple CSV files
        for platform in [1, 2, 3]:
            for video in [1, 2, 3]:
                for session in [1, 2]:
                    filename = f"{platform}_{video}_{session}_{user_dir.name}.csv"
                    filepath = user_dir / filename
                    # Create headerless CSV
                    with open(filepath, 'w') as f:
                        f.write("P,a,100\n")
                        f.write("R,a,150\n")
                        f.write("P,b,200\n")
                        f.write("R,b,250\n")
        
        # Update version manager to simulate clean stage completion
        vm = VersionManager()
        vm.register_version(test_version_id, {"test": True})
        vm.update_stage_info(test_version_id, "clean_data", {
            "output_path": str(artifacts_dir / "cleaned_data"),
            "completed": True
        })
        
        # Run keypairs stage
        output_dir = extract_keypairs.run(
            version_id=test_version_id,
            config=test_config,
            dry_run=False,
            local_only=True
        )
        
        # Verify output
        assert output_dir.exists()
        assert output_dir.name == "keypairs"
        
        # Check for output files
        parquet_file = output_dir / "keypairs.parquet"
        csv_file = output_dir / "keypairs.csv"
        
        assert parquet_file.exists()
        assert csv_file.exists()
        
        # Load and verify data
        import pandas as pd
        df = pd.read_parquet(parquet_file)
        
        assert len(df) > 0
        assert "user_id" in df.columns
        assert "HL" in df.columns
        assert "IL" in df.columns
        assert "valid" in df.columns
    
    def test_features_generates_from_keypairs(self, test_version_id, test_config):
        """Test that features stage generates features from keypairs"""
        # Setup keypairs data
        artifacts_dir = Path(test_config["ARTIFACTS_DIR"]) / test_version_id
        keypairs_dir = artifacts_dir / "keypairs"
        keypairs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample keypairs data
        import pandas as pd
        import numpy as np
        
        keypairs_data = pd.DataFrame({
            'user_id': ['user1'] * 100,
            'platform_id': [1] * 100,
            'video_id': [1] * 100,
            'session_id': [1] * 100,
            'key1': np.random.choice(['a', 'b', 'c'], 100),
            'key2': np.random.choice(['a', 'b', 'c'], 100),
            'HL': np.random.normal(100, 20, 100),
            'IL': np.random.normal(50, 10, 100),
            'PL': np.random.normal(80, 15, 100),
            'RL': np.random.normal(120, 25, 100),
            'valid': True,
            'outlier': False,
            'device_type': 'desktop'
        })
        
        keypairs_data.to_parquet(keypairs_dir / "keypairs.parquet")
        
        # Update version manager
        vm = VersionManager()
        vm.register_version(test_version_id, {"test": True})
        vm.update_stage_info(test_version_id, "extract_keypairs", {
            "output_path": str(keypairs_dir),
            "completed": True
        })
        
        # Run features stage
        output_dir = extract_features.run(
            version_id=test_version_id,
            config=test_config,
            dry_run=False,
            local_only=True
        )
        
        # Verify output
        assert output_dir.exists()
        assert output_dir.name == "features"
        
        # Check for feature directories
        expected_dirs = [
            "typenet_ml_user_platform",
            "typenet_ml_session",
            "typenet_ml_video"
        ]
        
        for dir_name in expected_dirs:
            feature_dir = output_dir / dir_name
            assert feature_dir.exists()
            
            # Check for feature files
            feature_file = feature_dir / "features.parquet"
            assert feature_file.exists()
            
            # Verify data
            df = pd.read_parquet(feature_file)
            assert len(df) > 0
            assert "user_id" in df.columns
    
    def test_full_pipeline_data_flow(self, test_version_id, test_config):
        """Test complete data flow through all stages"""
        # Create test data
        test_data_dir = Path("test_data/raw_data")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        generator = FakeDataGenerator()
        # Create multiple users
        complete_user_id = generator.generate_complete_user(test_data_dir, "complete")
        
        incomplete_user_id = generator.generate_user_files(test_data_dir, "incomplete", num_files=10)
        
        try:
            # Register version for the full pipeline test
            vm = VersionManager()
            vm.register_version(test_version_id, {"test": True})
            
            # Stage 1: Download
            download_output = download_data.run(
                version_id=test_version_id,
                config=test_config,
                dry_run=False,
                local_only=True
            )
            assert download_output.exists()
            assert len(list(download_output.glob("*"))) > 0
            
            # Stage 2: Clean
            clean_output = clean_data.run(
                version_id=test_version_id,
                config=test_config,
                dry_run=False,
                local_only=True
            )
            assert clean_output.exists()
            
            # Verify user organization
            desktop_raw = clean_output / "desktop" / "raw_data"
            desktop_broken = clean_output / "desktop" / "broken_data"
            
            assert (desktop_raw / complete_user_id).exists()
            assert (desktop_broken / incomplete_user_id).exists()
            
            # Stage 3: Keypairs
            keypairs_output = extract_keypairs.run(
                version_id=test_version_id,
                config=test_config,
                dry_run=False,
                local_only=True
            )
            assert keypairs_output.exists()
            assert (keypairs_output / "keypairs.parquet").exists()
            
            # Stage 4: Features
            features_output = extract_features.run(
                version_id=test_version_id,
                config=test_config,
                dry_run=False,
                local_only=True
            )
            assert features_output.exists()
            
            # Verify final output structure
            artifacts_dir = Path(test_config["ARTIFACTS_DIR"]) / test_version_id
            
            # Check all expected directories exist
            assert (artifacts_dir / "raw_data" / "web_app_data").exists()
            assert (artifacts_dir / "cleaned_data" / "desktop" / "raw_data").exists()
            assert (artifacts_dir / "cleaned_data" / "desktop" / "broken_data").exists()
            assert (artifacts_dir / "keypairs").exists()
            assert (artifacts_dir / "features").exists()
            
            # Check metadata
            assert (artifacts_dir / "etl_metadata").exists()
            
        finally:
            # Cleanup
            shutil.rmtree(test_data_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])