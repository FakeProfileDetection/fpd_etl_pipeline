#!/usr/bin/env python3
"""
Extract Keypairs Stage
Extracts keystroke pairs and timing features from raw keystroke data

This stage:
- Processes raw keystroke CSV files (Press/Release events)
- Matches press-release pairs for individual keys
- Extracts consecutive key pairs
- Calculates TypeNet timing features (HL, IL, PL, RL)
- Validates data and identifies errors
- Saves extraction metadata in etl_metadata/keypairs/
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import EnhancedVersionManager as VersionManager

logger = logging.getLogger(__name__)


@dataclass
class KeystrokeEvent:
    """Represents a single keystroke event"""
    key: str
    press_time: Optional[float]
    release_time: Optional[float]
    press_idx: Optional[int]
    release_idx: Optional[int]
    valid: bool
    error: str


@dataclass
class KeypairFeatures:
    """TypeNet features for a key pair"""
    user_id: str
    platform_id: int
    video_id: int
    session_id: int
    sequence_id: int
    key1: str
    key2: str
    key1_press: Optional[float]
    key1_release: Optional[float]
    key2_press: Optional[float]
    key2_release: Optional[float]
    HL: Optional[float]  # Hold Latency (key1_release - key1_press)
    IL: Optional[float]  # Inter-key Latency (key2_press - key1_release)
    PL: Optional[float]  # Press Latency (key2_press - key1_press)
    RL: Optional[float]  # Release Latency (key2_release - key1_release)
    valid: bool
    error_description: str
    key1_timestamp: float  # For sorting


class ExtractKeypairsStage:
    """Extract keystroke pairs and timing features"""
    
    ERROR_TYPES = {
        'valid': 'No error',
        'missing_key1_release': 'Missing key1 release',
        'missing_key1_press': 'Missing key1 press (orphan release)',
        'missing_key2_release': 'Missing key2 release',
        'missing_key2_press': 'Missing key2 press (orphan release)',
        'negative_HL': 'Negative Hold Latency',
        'negative_PL': 'Negative Press Latency',
        'file_read_error': 'Error reading file',
        'invalid_format': 'Invalid file format'
    }
    
    def __init__(self, version_id: str, config: Dict[str, Any], 
                 dry_run: bool = False, local_only: bool = False,
                 version_manager: Optional[VersionManager] = None):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.version_manager = version_manager or VersionManager()
        
        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "total_keypairs": 0,
            "valid_keypairs": 0,
            "invalid_keypairs": 0,
            "error_counts": defaultdict(int),
            "user_stats": {},
            "processing_errors": []
        }
        
    def is_valid_raw_file(self, filepath: Path) -> bool:
        """
        Check if file is a raw keystroke data file
        Expected format: platform_video_session_user.csv (e.g., 1_1_1_3741e927ab7d45a7ca19ed47a3eb5864.csv)
        """
        if not filepath.suffix == '.csv':
            return False
            
        # Check filename pattern
        parts = filepath.stem.split('_')
        if len(parts) != 4:
            return False
            
        # First 3 parts should be numeric
        try:
            platform_id = int(parts[0])
            video_id = int(parts[1])
            session_id = int(parts[2])
            # 4th part is user ID (hash)
            return len(parts[3]) == 32  # MD5 hash length
        except ValueError:
            return False
            
    def parse_raw_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Parse raw keystroke file with format: press-type (P or R), key, timestamp"""
        try:
            # First check if file has header
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            # Determine if file has header by checking first line
            has_header = 'Press' in first_line or 'Time' in first_line
            
            # Read CSV with or without header
            if has_header:
                # Read with header, skip first row, then assign our names
                df = pd.read_csv(
                    filepath,
                    skiprows=1,
                    header=None,
                    names=['type', 'key', 'timestamp'],
                    dtype={'type': str, 'key': str, 'timestamp': float}
                )
            else:
                df = pd.read_csv(
                    filepath,
                    header=None,
                    names=['type', 'key', 'timestamp'],
                    dtype={'type': str, 'key': str, 'timestamp': float}
                )
            
            # Validate format
            if len(df.columns) != 3:
                logger.warning(f"Invalid format in {filepath.name}: wrong number of columns")
                return None
                
            # Check if type column contains expected values
            if not df['type'].isin(['P', 'R']).all():
                logger.warning(f"Invalid format in {filepath.name}: unexpected press types")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error reading {filepath.name}: {e}")
            self.stats["processing_errors"].append(f"Read error in {filepath.name}: {str(e)}")
            return None
            
    def match_press_release_pairs(self, df: pd.DataFrame) -> List[KeystrokeEvent]:
        """
        Match press and release events using a stack-based algorithm
        Returns list of keystroke events with validity information
        """
        events = []
        key_stacks = {}  # Stack for each unique key
        
        for idx, row in df.iterrows():
            key = row['key']
            event_type = row['type']
            timestamp = row['timestamp']
            
            if key not in key_stacks:
                key_stacks[key] = deque()
                
            if event_type == 'P':
                # Push press event to stack
                key_stacks[key].append({
                    'key': key,
                    'press_time': timestamp,
                    'press_idx': idx
                })
                
            elif event_type == 'R':
                if key_stacks[key]:
                    # Match with most recent press
                    press_event = key_stacks[key].pop()
                    events.append(KeystrokeEvent(
                        key=key,
                        press_time=press_event['press_time'],
                        release_time=timestamp,
                        press_idx=press_event['press_idx'],
                        release_idx=idx,
                        valid=True,
                        error='valid'
                    ))
                else:
                    # Orphan release - no matching press
                    events.append(KeystrokeEvent(
                        key=key,
                        press_time=None,
                        release_time=timestamp,
                        press_idx=None,
                        release_idx=idx,
                        valid=False,
                        error='missing_key1_press'
                    ))
                    
        # Handle unmatched press events (missing releases)
        for key, stack in key_stacks.items():
            while stack:
                press_event = stack.pop()
                events.append(KeystrokeEvent(
                    key=press_event['key'],
                    press_time=press_event['press_time'],
                    release_time=None,
                    press_idx=press_event['press_idx'],
                    release_idx=None,
                    valid=False,
                    error='missing_key1_release'
                ))
                
        # Sort events by press time (or release time for orphan releases)
        events.sort(key=lambda x: x.press_time if x.press_time is not None else x.release_time)
        
        return events
        
    def calculate_features(self, key1: KeystrokeEvent, key2: KeystrokeEvent, 
                         user_id: str, platform_id: int, video_id: int, 
                         session_id: int, sequence_id: int) -> KeypairFeatures:
        """Calculate TypeNet features for a key pair"""
        features = KeypairFeatures(
            user_id=user_id,
            platform_id=platform_id,
            video_id=video_id,
            session_id=session_id,
            sequence_id=sequence_id,
            key1=key1.key,
            key2=key2.key,
            key1_press=key1.press_time,
            key1_release=key1.release_time,
            key2_press=key2.press_time,
            key2_release=key2.release_time,
            HL=None,
            IL=None,
            PL=None,
            RL=None,
            valid=True,
            error_description='No error',
            key1_timestamp=key1.press_time if key1.press_time is not None else key1.release_time
        )
        
        # Check validity
        if not key1.valid:
            features.valid = False
            features.error_description = self.ERROR_TYPES.get(key1.error, key1.error)
            
        if not key2.valid:
            features.valid = False
            if features.error_description == 'No error':
                features.error_description = self.ERROR_TYPES.get(key2.error, key2.error).replace('key1', 'key2')
            else:
                features.error_description += f"; {self.ERROR_TYPES.get(key2.error, key2.error).replace('key1', 'key2')}"
                
        # Calculate HL for key1 (can be calculated even if key2 is invalid)
        if key1.press_time is not None and key1.release_time is not None:
            features.HL = key1.release_time - key1.press_time
            if features.HL < 0:
                features.valid = False
                features.error_description = 'Negative Hold Latency'
                
        # Calculate other features only if both keys are valid
        if key1.valid and key2.valid:
            try:
                # IL: Inter-key Latency (can be negative for overlapping keys)
                if key1.release_time is not None and key2.press_time is not None:
                    features.IL = key2.press_time - key1.release_time
                    
                # PL: Press Latency
                if key1.press_time is not None and key2.press_time is not None:
                    features.PL = key2.press_time - key1.press_time
                    if features.PL < 0:
                        features.valid = False
                        features.error_description = 'Negative Press Latency'
                        
                # RL: Release Latency (can be negative)
                if key1.release_time is not None and key2.release_time is not None:
                    features.RL = key2.release_time - key1.release_time
                    
            except Exception as e:
                features.valid = False
                features.error_description = f'Calculation error: {str(e)}'
                
        return features
        
    def extract_features_from_file(self, filepath: Path) -> List[KeypairFeatures]:
        """Extract all features from a single raw keystroke file"""
        # Parse filename to get metadata
        parts = filepath.stem.split('_')
        platform_id = int(parts[0])
        video_id = int(parts[1])
        session_id = int(parts[2])
        user_id = parts[3]
        
        # Read and parse raw data
        df = self.parse_raw_file(filepath)
        if df is None:
            self.stats["error_counts"]["file_read_error"] += 1
            return []
            
        # Match press-release pairs
        events = self.match_press_release_pairs(df)
        
        # Extract features for consecutive key pairs
        features_list = []
        for i in range(len(events) - 1):
            features = self.calculate_features(
                events[i], events[i + 1],
                user_id, platform_id, video_id, session_id, i
            )
            features_list.append(features)
            
            # Update error statistics
            if not features.valid:
                self.stats["error_counts"][features.error_description] += 1
                
        return features_list
        
    def process_user_directory(self, user_dir: Path) -> pd.DataFrame:
        """Process all keystroke files for a single user"""
        user_id = user_dir.name
        all_features = []
        
        # Find all raw keystroke files
        csv_files = [f for f in user_dir.glob("*.csv") if self.is_valid_raw_file(f)]
        
        for csv_file in sorted(csv_files):
            self.stats["total_files"] += 1
            
            try:
                features = self.extract_features_from_file(csv_file)
                all_features.extend(features)
                self.stats["processed_files"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                self.stats["processing_errors"].append(f"Process error in {csv_file.name}: {str(e)}")
                self.stats["skipped_files"] += 1
                
        # Convert to DataFrame
        if all_features:
            df = pd.DataFrame([asdict(f) for f in all_features])
            
            # Add outlier detection
            df['outlier'] = False
            if len(df) > 0 and df['valid'].any():
                # Mark as outliers if timing is extreme (only for valid keypairs)
                # Note: timings are in milliseconds, not nanoseconds
                valid_mask = df['valid']
                df.loc[valid_mask & (df['HL'] > 2000), 'outlier'] = True  # > 2 seconds
                df.loc[valid_mask & (df['HL'] < 30), 'outlier'] = True   # < 30ms
                df.loc[valid_mask & (df['IL'].abs() > 1000), 'outlier'] = True  # |IL| > 1 second
            
            # Calculate user statistics
            valid_count = df['valid'].sum()
            total_count = len(df)
            self.stats["user_stats"][user_id] = {
                "total_keypairs": total_count,
                "valid_keypairs": valid_count,
                "invalid_keypairs": total_count - valid_count,
                "validity_rate": (valid_count / total_count * 100) if total_count > 0 else 0
            }
            
            return df
        else:
            # Return empty dataframe with correct schema
            return pd.DataFrame(columns=[
                'user_id', 'platform_id', 'session_id', 'video_id', 'sequence_id',
                'key1', 'key2', 'key1_press', 'key1_release', 'key2_press', 'key2_release',
                'HL', 'IL', 'PL', 'RL', 'valid', 'error_description', 'key1_timestamp', 'outlier'
            ])
            
    def run(self, input_dir: Path) -> Path:
        """Execute the extract keypairs stage"""
        logger.info(f"Starting Extract Keypairs stage for version {self.version_id}")
        logger.info(f"Input directory: {input_dir}")
        
        # Setup output directories following artifacts structure
        artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        output_dir = artifacts_dir / "keypairs"
        metadata_dir = artifacts_dir / "etl_metadata" / "keypairs"
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        # Process configured device types
        all_keypair_data = []
        
        # Get device types from config (default: ["desktop"])
        from scripts.utils.config_manager import get_config
        config_manager = get_config()
        device_types = config_manager.get_device_types()
        logger.info(f"Processing device types: {device_types}")
        
        for device_type in device_types:
            device_dir = input_dir / device_type / "raw_data"
            if not device_dir.exists():
                logger.info(f"No {device_type} data found, skipping")
                continue
                
            logger.info(f"Processing {device_type} data...")
            
            # Process each user
            for user_dir in device_dir.iterdir():
                if user_dir.is_dir():
                    user_df = self.process_user_directory(user_dir)
                    if not user_df.empty:
                        user_df['device_type'] = device_type
                        all_keypair_data.append(user_df)
                        
        # Combine all data
        if all_keypair_data:
            combined_df = pd.concat(all_keypair_data, ignore_index=True)
            
            # Update overall statistics
            self.stats["total_keypairs"] = len(combined_df)
            self.stats["valid_keypairs"] = combined_df['valid'].sum()
            self.stats["invalid_keypairs"] = self.stats["total_keypairs"] - self.stats["valid_keypairs"]
        else:
            # Create empty dataframe with correct schema
            combined_df = pd.DataFrame(columns=[
                'user_id', 'platform_id', 'session_id', 'video_id', 'device_type',
                'key1', 'key2', 'HL', 'IL', 'PL', 'RL', 'valid', 'outlier'
            ])
            
        # Save keypair data (even if empty)
        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet for efficiency
            output_file = output_dir / "keypairs.parquet"
            combined_df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(combined_df)} keypairs to {output_file}")
            
            # Also save as CSV for compatibility
            csv_file = output_dir / "keypairs.csv"
            combined_df.to_csv(csv_file, index=False)
            
            # Save invalid sequences for analysis
            if not combined_df.empty:
                invalid_df = combined_df[~combined_df['valid']]
                if not invalid_df.empty:
                    invalid_file = metadata_dir / "invalid_sequences.csv"
                    metadata_dir.mkdir(parents=True, exist_ok=True)
                    invalid_df.to_csv(invalid_file, index=False)
                    
        # Save extraction metadata
        if not self.dry_run:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Extraction stats
            extraction_stats = {
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "summary": {
                    "total_files": int(self.stats["total_files"]),
                    "processed_files": int(self.stats["processed_files"]),
                    "skipped_files": int(self.stats["skipped_files"]),
                    "total_keypairs": int(self.stats["total_keypairs"]),
                    "valid_keypairs": int(self.stats["valid_keypairs"]),
                    "invalid_keypairs": int(self.stats["invalid_keypairs"]),
                    "validity_rate": float((self.stats["valid_keypairs"] / self.stats["total_keypairs"] * 100) 
                                   if self.stats["total_keypairs"] > 0 else 0)
                },
                "error_distribution": {k: int(v) for k, v in self.stats["error_counts"].items()},
                "user_count": len(self.stats["user_stats"])
            }
            
            with open(metadata_dir / "extraction_stats.json", 'w') as f:
                json.dump(extraction_stats, f, indent=2)
                
            # Processing log
            if self.stats["processing_errors"]:
                with open(metadata_dir / "processing_log.json", 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "errors": self.stats["processing_errors"]
                    }, f, indent=2)
                    
        # Log summary
        logger.info(f"Extraction complete:")
        logger.info(f"  Total files: {self.stats['total_files']}")
        logger.info(f"  Processed files: {self.stats['processed_files']}")
        logger.info(f"  Total keypairs: {self.stats['total_keypairs']}")
        logger.info(f"  Valid keypairs: {self.stats['valid_keypairs']} "
                   f"({self.stats['valid_keypairs'] / self.stats['total_keypairs'] * 100:.1f}%)" 
                   if self.stats['total_keypairs'] > 0 else "")
        
        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                "extract_keypairs",
                {
                    "output_dir": str(output_dir),
                    "stats": extraction_stats["summary"],
                    "completed_at": datetime.now().isoformat()
                }
            )
            
        return output_dir


def run(version_id: str, config: Dict[str, Any], 
        dry_run: bool = False, local_only: bool = False) -> Path:
    """Entry point for the pipeline orchestrator"""
    # Get input directory from previous stage
    vm = VersionManager()
    version_info = vm.get_version(version_id)
    
    if not version_info or "clean_data" not in version_info.get("stages", {}):
        # Default input directory
        artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts")) / version_id
        input_dir = artifacts_dir / "cleaned_data"
    else:
        # Handle both output_dir and output_path for compatibility
        clean_info = version_info["stages"]["clean_data"]
        path_key = "output_path" if "output_path" in clean_info else "output_dir"
        input_dir = Path(clean_info[path_key])
        
    stage = ExtractKeypairsStage(version_id, config, dry_run, local_only)
    return stage.run(input_dir)


if __name__ == "__main__":
    # For testing the stage independently
    import click
    from scripts.utils.config_manager import get_config
    
    @click.command()
    @click.option('--version-id', help='Version ID to use')
    @click.option('--input-dir', help='Input directory (overrides default)')
    @click.option('--dry-run', is_flag=True, help='Preview without processing')
    def main(version_id, input_dir, dry_run):
        """Test Extract Keypairs stage independently"""
        logging.basicConfig(level=logging.INFO)
        
        config = get_config()._config
        vm = VersionManager()
        
        if not version_id:
            version_id = vm.create_version_id()
            logger.info(f"Created version ID: {version_id}")
            
        if input_dir:
            stage = ExtractKeypairsStage(version_id, config, dry_run)
            output_dir = stage.run(Path(input_dir))
        else:
            output_dir = run(version_id, config, dry_run)
            
        logger.info(f"Stage complete. Output: {output_dir}")
        
    main()