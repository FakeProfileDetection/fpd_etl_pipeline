"""Data validation utilities for pipeline stages"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KeystrokeDataValidator:
    """Validator for keystroke data quality"""
    
    def __init__(self):
        self.skip_keys = {'Key.shift', 'Key.ctrl', 'Key.alt', 'Key.cmd', 'Key.caps_lock'}
        
    def validate_csv_format(self, csv_path: Path) -> Tuple[bool, List[str]]:
        """Validate CSV file format"""
        errors = []
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_cols = ['Press or Release', 'Key', 'Time']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
                return False, errors
                
            # Check data types
            if not pd.api.types.is_numeric_dtype(df['Time']):
                errors.append("Time column must be numeric")
                
            # Check event types
            valid_events = {'P', 'R'}
            invalid_events = set(df['Press or Release'].unique()) - valid_events
            if invalid_events:
                errors.append(f"Invalid event types: {invalid_events}")
                
            # Check for empty data
            if len(df) == 0:
                errors.append("CSV file is empty")
                
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error reading CSV: {str(e)}")
            return False, errors
            
    def check_timing_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for timing inconsistencies in keystroke data"""
        issues = {
            'orphan_releases': [],
            'double_presses': [],
            'negative_hold_times': [],
            'unreleased_keys': {},
            'timing_jumps': []
        }
        
        # Sort by time
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Track active keys
        active_keys = {}
        
        for idx, row in df.iterrows():
            key = row['Key']
            event_type = row['Press or Release']
            time = row['Time']
            
            # Skip modifier keys for some checks
            if key in self.skip_keys:
                continue
                
            if event_type == 'P':
                if key in active_keys:
                    # Double press
                    issues['double_presses'].append({
                        'key': key,
                        'first_press': active_keys[key],
                        'second_press': time,
                        'index': idx
                    })
                active_keys[key] = time
                
            elif event_type == 'R':
                if key not in active_keys:
                    # Orphan release
                    issues['orphan_releases'].append({
                        'key': key,
                        'time': time,
                        'index': idx
                    })
                else:
                    # Check hold time
                    hold_time = time - active_keys[key]
                    if hold_time < 0:
                        issues['negative_hold_times'].append({
                            'key': key,
                            'press_time': active_keys[key],
                            'release_time': time,
                            'hold_time': hold_time,
                            'index': idx
                        })
                    del active_keys[key]
                    
        # Check for timing jumps
        if len(df) > 1:
            time_diffs = df['Time'].diff()
            large_jumps = time_diffs[time_diffs > 10000]  # > 10 seconds
            
            for idx in large_jumps.index:
                if idx > 0:
                    issues['timing_jumps'].append({
                        'index': idx,
                        'jump_ms': time_diffs[idx],
                        'from_time': df.loc[idx-1, 'Time'],
                        'to_time': df.loc[idx, 'Time']
                    })
                    
        # Record unreleased keys
        issues['unreleased_keys'] = {k: v for k, v in active_keys.items() 
                                   if k not in self.skip_keys}
        
        return issues
        
        
class UserDataValidator:
    """Validator for user data completeness"""
    
    @staticmethod
    def check_user_completeness(user_dir: Path) -> Dict[str, Any]:
        """Check if user has all required files"""
        result = {
            'complete': True,
            'missing_required': [],
            'missing_optional': [],
            'files_found': []
        }
        
        user_id = user_dir.name
        
        # Required files
        required_files = [
            f"{user_id}_consent.json",
            f"{user_id}_demographics.json",
            f"{user_id}_start_time.json"
        ]
        
        # Optional files
        optional_files = [
            f"{user_id}_completion.json"
        ]
        
        # Check required files
        for req_file in required_files:
            if not (user_dir / req_file).exists():
                result['missing_required'].append(req_file)
                result['complete'] = False
            else:
                result['files_found'].append(req_file)
                
        # Check optional files
        for opt_file in optional_files:
            if not (user_dir / opt_file).exists():
                result['missing_optional'].append(opt_file)
            else:
                result['files_found'].append(opt_file)
                
        # Check for typing data
        typing_files = list(user_dir.glob(f"[fit]_{user_id}_*.csv"))
        result['typing_file_count'] = len(typing_files)
        
        # Need at least one typing file
        if len(typing_files) == 0:
            result['complete'] = False
            result['missing_required'].append("No typing data files found")
            
        return result
        

class FeatureDataValidator:
    """Validator for extracted features"""
    
    @staticmethod
    def validate_feature_dataframe(df: pd.DataFrame, feature_type: str) -> Dict[str, Any]:
        """Validate feature dataframe"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for required ID columns based on feature type
        id_columns = {
            'user_platform': ['user_id', 'platform_id'],
            'session': ['user_id', 'platform_id', 'session_id'],
            'video': ['user_id', 'platform_id', 'session_id', 'video_id']
        }
        
        # Determine feature type from name
        if 'video' in feature_type:
            required_ids = id_columns['video']
        elif 'session' in feature_type:
            required_ids = id_columns['session']
        else:
            required_ids = id_columns['user_platform']
            
        # Check ID columns
        missing_ids = [col for col in required_ids if col not in df.columns]
        if missing_ids:
            result['errors'].append(f"Missing ID columns: {missing_ids}")
            result['valid'] = False
            
        # Check for NaN values
        nan_counts = df.isna().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            result['warnings'].append(f"Found {total_nans} NaN values")
            result['stats']['nan_columns'] = nan_counts[nan_counts > 0].to_dict()
            
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        total_infs = inf_counts.sum()
        
        if total_infs > 0:
            result['errors'].append(f"Found {total_infs} infinite values")
            result['valid'] = False
            
        # Check value ranges
        for col in numeric_cols:
            if col.startswith('HL_') or col.startswith('IL_'):
                # Timing features should be non-negative (except IL can be negative)
                if col.startswith('HL_'):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        result['warnings'].append(
                            f"{col}: {negative_count} negative hold latencies"
                        )
                        
        # Calculate basic statistics
        result['stats']['shape'] = df.shape
        result['stats']['memory_usage'] = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        return result
        

def validate_pipeline_output(version_id: str, artifacts_dir: Path) -> Dict[str, Any]:
    """Validate the complete pipeline output for a version"""
    results = {
        'version_id': version_id,
        'valid': True,
        'stages': {}
    }
    
    version_dir = artifacts_dir / version_id
    
    # Check cleaned data
    cleaned_dir = version_dir / "cleaned_data"
    if cleaned_dir.exists():
        user_validator = UserDataValidator()
        clean_stats = {
            'desktop_users': 0,
            'mobile_users': 0,
            'total_complete': 0,
            'total_broken': 0
        }
        
        for device in ['desktop', 'mobile']:
            device_dir = cleaned_dir / device
            if device_dir.exists():
                # Count users
                raw_users = list((device_dir / "raw_data").iterdir()) if (device_dir / "raw_data").exists() else []
                broken_users = list((device_dir / "broken_data").iterdir()) if (device_dir / "broken_data").exists() else []
                
                clean_stats[f'{device}_users'] = len(raw_users)
                clean_stats['total_complete'] += len(raw_users)
                clean_stats['total_broken'] += len(broken_users)
                
        results['stages']['clean_data'] = clean_stats
    else:
        results['stages']['clean_data'] = {'error': 'Output directory not found'}
        results['valid'] = False
        
    # Check keypairs
    keypair_file = version_dir / "keypairs" / "keypairs.parquet"
    if not keypair_file.exists():
        keypair_file = version_dir / "keypairs" / "keypairs.csv"
        
    if keypair_file.exists():
        try:
            df = pd.read_parquet(keypair_file) if keypair_file.suffix == '.parquet' else pd.read_csv(keypair_file)
            results['stages']['extract_keypairs'] = {
                'total_keypairs': len(df),
                'valid_keypairs': df['valid'].sum() if 'valid' in df.columns else 0,
                'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0
            }
        except Exception as e:
            results['stages']['extract_keypairs'] = {'error': str(e)}
            results['valid'] = False
    else:
        results['stages']['extract_keypairs'] = {'error': 'Output file not found'}
        results['valid'] = False
        
    # Check features
    features_dir = version_dir / "features"
    if features_dir.exists():
        feature_validator = FeatureDataValidator()
        feature_results = {}
        
        for feature_type_dir in features_dir.iterdir():
            if feature_type_dir.is_dir():
                feature_file = feature_type_dir / "features.parquet"
                if not feature_file.exists():
                    feature_file = feature_type_dir / "features.csv"
                    
                if feature_file.exists():
                    try:
                        df = pd.read_parquet(feature_file) if feature_file.suffix == '.parquet' else pd.read_csv(feature_file)
                        validation = feature_validator.validate_feature_dataframe(df, feature_type_dir.name)
                        feature_results[feature_type_dir.name] = validation
                    except Exception as e:
                        feature_results[feature_type_dir.name] = {'error': str(e)}
                        
        results['stages']['extract_features'] = feature_results
    else:
        results['stages']['extract_features'] = {'error': 'Output directory not found'}
        
    return results