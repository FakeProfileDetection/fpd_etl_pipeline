"""Test utilities and fixtures for pipeline tests"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class TestDataGenerator:
    """Generate test data for pipeline stages"""
    
    def __init__(self):
        # Use valid 32-character hex strings (like MD5 hashes)
        self.test_user_ids = [
            "a1b2c3d4e5f6789012345678901234ef",
            "b2c3d4e5f67890123456789012345678",
            "c3d4e5f678901234567890123456789a"
        ]
        
    def create_keystroke_csv(self, output_path: Path, num_events: int = 100,
                           include_errors: bool = False) -> None:
        """Create a test keystroke CSV file"""
        events = []
        keys = ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd', 'Key.shift', 'Key.space']
        current_time = int(datetime.now().timestamp() * 1000)
        
        # Track active keys for realistic data
        active_keys = {}
        
        for i in range(num_events):
            key = random.choice(keys)
            
            if key not in active_keys or random.random() > 0.7:
                # Press event
                event_type = 'P'
                active_keys[key] = current_time
            else:
                # Release event
                event_type = 'R'
                del active_keys[key]
                
            events.append([event_type, key, current_time])
            
            # Add some time between events
            current_time += random.randint(10, 200)
            
        # Add data quality issues if requested
        if include_errors:
            # Orphan release
            events.append(['R', 'x', current_time + 100])
            
            # Double press
            events.append(['P', 'y', current_time + 200])
            events.append(['P', 'y', current_time + 300])
            
        # Save to CSV without headers (as expected by extract_keypairs)
        df = pd.DataFrame(events)
        df.to_csv(output_path, index=False, header=False)
        
    def create_metadata_json(self, output_path: Path, metadata_type: str = 'text') -> None:
        """Create a test metadata JSON file"""
        metadata = {
            'type': metadata_type,
            'platform': random.choice([1, 2, 3]),
            'session': random.choice([1, 2]),
            'video': random.choice([1, 2, 3, 4, 5, 6]),
            'user': random.choice(self.test_user_ids),
            'timestamp': datetime.now().isoformat(),
            'duration_ms': random.randint(5000, 30000)
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def create_raw_text(self, output_path: Path, text_type: str = 'free') -> None:
        """Create a test raw text file"""
        texts = {
            'free': "Hello world, this is a test typing sample.",
            'transcription': "The quick brown fox jumps over the lazy dog.",
            'image': "A beautiful sunset over the ocean with orange and pink colors."
        }
        
        text = texts.get(text_type, texts['free'])
        output_path.write_text(text)
        
    def create_user_info_json(self, output_path: Path, info_type: str) -> None:
        """Create user info JSON files (consent, demographics, etc.)"""
        data = {}
        
        if info_type == 'consent':
            data = {
                'consent': True,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
        elif info_type == 'demographics':
            data = {
                'age': random.randint(18, 65),
                'gender': random.choice(['M', 'F', 'Other']),
                'education': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'typing_experience': random.choice(['Beginner', 'Intermediate', 'Advanced'])
            }
        elif info_type == 'start_time':
            data = {
                'start_time': datetime.now().isoformat(),
                'timezone': 'UTC'
            }
        elif info_type == 'completion':
            data = {
                'completed': True,
                'end_time': (datetime.now() + timedelta(minutes=30)).isoformat(),
                'videos_completed': 18
            }
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def create_complete_user_data(self, user_dir: Path, user_id: str,
                                include_all_files: bool = True) -> None:
        """Create a complete set of user data files"""
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Essential files
        self.create_user_info_json(user_dir / f"{user_id}_consent.json", 'consent')
        self.create_user_info_json(user_dir / f"{user_id}_demographics.json", 'demographics')
        self.create_user_info_json(user_dir / f"{user_id}_start_time.json", 'start_time')
        
        if include_all_files:
            self.create_user_info_json(user_dir / f"{user_id}_completion.json", 'completion')
            
        # Create typing data for each type and index
        # Need 6 videos of each type for complete data
        # Use the exact sequence numbers expected by clean_data stage
        sequences = {
            'f': [0, 3, 6, 9, 12, 15],     # Facebook 
            'i': [1, 4, 7, 10, 13, 16],    # Instagram
            't': [2, 5, 8, 11, 14, 17]     # Twitter
        }
        
        for task_type, prefix in [('free', 'f'), ('image', 'i'), ('transcription', 't')]:
            # For incomplete users, only create first 3 files of each platform
            seq_list = sequences[prefix] if include_all_files else sequences[prefix][:3]
            
            for seq_num in seq_list:
                # Keystroke data
                csv_file = user_dir / f"{prefix}_{user_id}_{seq_num}.csv"
                self.create_keystroke_csv(csv_file)
                
                # Metadata
                meta_file = user_dir / f"{prefix}_{user_id}_{seq_num}_metadata.json"
                self.create_metadata_json(meta_file, task_type)
                
                # Raw text
                raw_file = user_dir / f"{prefix}_{user_id}_{seq_num}_raw.txt"
                self.create_raw_text(raw_file, task_type)
                
        # Also create platform files for session/video structure
        for session in [1, 2]:
            for video in [1, 2, 3]:
                # Platform_Session_Video_UserID.csv format
                platform_file = user_dir / f"{1}_{session}_{video}_{user_id}.csv"
                if not platform_file.exists():
                    self.create_keystroke_csv(platform_file)
                
                
class TestValidator:
    """Validate test outputs from pipeline stages"""
    
    @staticmethod
    def validate_cleaned_data_structure(cleaned_dir: Path) -> Dict[str, Any]:
        """Validate the structure of cleaned data output"""
        results = {
            'valid': True,
            'errors': [],
            'stats': {}
        }
        
        # Check for required directories
        for device in ['desktop', 'mobile']:
            device_dir = cleaned_dir / device
            if not device_dir.exists():
                results['errors'].append(f"Missing {device} directory")
                results['valid'] = False
                continue
                
            # Check subdirectories
            for subdir in ['raw_data', 'broken_data', 'metadata', 'text']:
                if not (device_dir / subdir).exists():
                    results['errors'].append(f"Missing {device}/{subdir} directory")
                    results['valid'] = False
                    
            # Count users
            raw_users = len(list((device_dir / 'raw_data').iterdir())) if (device_dir / 'raw_data').exists() else 0
            broken_users = len(list((device_dir / 'broken_data').iterdir())) if (device_dir / 'broken_data').exists() else 0
            
            results['stats'][device] = {
                'complete_users': raw_users,
                'broken_users': broken_users
            }
            
        return results
        
    @staticmethod
    def validate_keypair_data(keypair_file: Path) -> Dict[str, Any]:
        """Validate keypair extraction output"""
        results = {
            'valid': True,
            'errors': [],
            'stats': {}
        }
        
        if not keypair_file.exists():
            results['errors'].append("Keypair file does not exist")
            results['valid'] = False
            return results
            
        # Load and check data
        try:
            if keypair_file.suffix == '.parquet':
                df = pd.read_parquet(keypair_file)
            else:
                df = pd.read_csv(keypair_file)
                
            # Check required columns
            required_cols = ['user_id', 'key1', 'key2', 'HL', 'IL', 'PL', 'RL', 'valid']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                results['errors'].append(f"Missing columns: {missing_cols}")
                results['valid'] = False
                
            # Calculate statistics
            results['stats'] = {
                'total_keypairs': len(df),
                'valid_keypairs': df['valid'].sum() if 'valid' in df.columns else 0,
                'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
                'unique_key1': df['key1'].nunique() if 'key1' in df.columns else 0,
                'unique_key2': df['key2'].nunique() if 'key2' in df.columns else 0
            }
            
            # Check for timing anomalies
            if 'HL' in df.columns:
                negative_hl = (df['HL'] < 0).sum()
                if negative_hl > 0:
                    results['errors'].append(f"{negative_hl} negative hold latencies found")
                    
        except Exception as e:
            results['errors'].append(f"Error reading keypair file: {str(e)}")
            results['valid'] = False
            
        return results
        
    @staticmethod
    def validate_features(feature_dir: Path) -> Dict[str, Any]:
        """Validate feature extraction output"""
        results = {
            'valid': True,
            'errors': [],
            'stats': {}
        }
        
        if not feature_dir.exists():
            results['errors'].append("Feature directory does not exist")
            results['valid'] = False
            return results
            
        # Check for feature files
        feature_types = list(feature_dir.iterdir())
        if not feature_types:
            results['errors'].append("No feature types found")
            results['valid'] = False
            return results
            
        for feature_type_dir in feature_types:
            if not feature_type_dir.is_dir():
                continue
                
            feature_file = feature_type_dir / "features.parquet"
            if not feature_file.exists():
                feature_file = feature_type_dir / "features.csv"
                
            if feature_file.exists():
                try:
                    if feature_file.suffix == '.parquet':
                        df = pd.read_parquet(feature_file)
                    else:
                        df = pd.read_csv(feature_file)
                        
                    # Check for NaN values
                    nan_counts = df.isna().sum()
                    total_nans = nan_counts.sum()
                    
                    results['stats'][feature_type_dir.name] = {
                        'shape': df.shape,
                        'total_nans': int(total_nans),
                        'columns_with_nans': int((nan_counts > 0).sum())
                    }
                    
                    if total_nans > 0:
                        results['errors'].append(
                            f"{feature_type_dir.name}: {total_nans} NaN values after imputation"
                        )
                        
                except Exception as e:
                    results['errors'].append(
                        f"Error reading {feature_type_dir.name}: {str(e)}"
                    )
                    
        return results


def create_test_config() -> Dict[str, Any]:
    """Create a test configuration"""
    return {
        "ARTIFACTS_DIR": "test_artifacts",
        "RAW_DATA_DIR": "test_artifacts/{version_id}/raw_data",
        "CLEANED_DATA_DIR": "test_artifacts/{version_id}/cleaned_data",
        "KEYPAIRS_DIR": "test_artifacts/{version_id}/keypairs",
        "FEATURES_DIR": "test_artifacts/{version_id}/features",
        "UPLOAD_ARTIFACTS": False,
        "INCLUDE_PII": False,
        "GENERATE_REPORTS": True,
        "KEEP_OUTLIERS": False,
        "BUCKET_NAME": "test-bucket",
        "PROJECT_ID": "test-project"
    }


def setup_test_version_manager(test_dir: Path):
    """Setup a test-specific version manager that uses a temp directory"""
    import sys
    # Add project root to path if not already there
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from scripts.utils.version_manager import VersionManager
    
    # Create test version file
    test_version_file = test_dir / "test_versions.json"
    
    # Initialize test versions file with empty data
    test_versions_data = {
        "versions": [],
        "current": None,
        "schema_version": "1.0"
    }
    with open(test_version_file, 'w') as f:
        json.dump(test_versions_data, f)
    
    # Create a test version manager instance directly
    test_vm = VersionManager(test_version_file)
    
    # Return the test instance
    return test_vm


def cleanup_test_version_manager():
    """Cleanup function kept for compatibility"""
    # No longer needed since we're not monkey-patching
    pass