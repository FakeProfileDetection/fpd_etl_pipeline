#!/usr/bin/env python3
"""
Map web app keystroke data to user-centric directory structure.

This script processes raw data collected from a web application and reorganizes it
into a format compatible with the TypeNet feature extraction pipeline. It handles
device type separation, metadata extraction, and data validation.

Input structure: Raw files with format {platform}_{user_hash}_{sequence}.csv
Output structure: 
  - CSV files: {device_type}/raw_data/{user_hash}/platform_video_session_user.csv
  - Text files: {device_type}/text/{user_hash}/platform_video_session_user.txt
"""

import os
import shutil
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import hashlib
import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
import re


@dataclass
class UserMetadata:
    """Stores user demographic and device information."""
    user_id: str
    email: str
    gender: str
    handedness: str
    education: str
    consent_email_for_future: bool
    submission_timestamp: str
    is_mobile: bool
    device_type: str
    user_agent: str
    screen_width: int
    screen_height: int
    platform: str
    
    @classmethod
    def from_demographics(cls, user_id: str, demo_data: Dict) -> 'UserMetadata':
        """Create UserMetadata from demographics JSON."""
        device_info = demo_data.get('device_info', {})
        return cls(
            user_id=user_id,
            email=demo_data.get('email', ''),
            gender=demo_data.get('gender', ''),
            handedness=demo_data.get('handedness', ''),
            education=demo_data.get('education', ''),
            consent_email_for_future=demo_data.get('consent_email_for_future', False),
            submission_timestamp=demo_data.get('submission_timestamp', ''),
            is_mobile=demo_data.get('is_mobile', False),
            device_type=demo_data.get('device_type', 'unknown'),
            user_agent=device_info.get('userAgent', ''),
            screen_width=device_info.get('screenWidth', 0),
            screen_height=device_info.get('screenHeight', 0),
            platform=device_info.get('platform', '')
        )


class DataIntegrityError(Exception):
    """Custom exception for data integrity issues."""
    pass


class WebAppDataProcessor:
    """Processes web app data into TypeNet-compatible format."""
    
    # Platform mapping
    PLATFORM_MAP = {
        'f': 1,  # Facebook
        'i': 2,  # Instagram
        't': 3   # Twitter
    }
    
    # Expected files per user  
    REQUIRED_FILES = {
        'metadata': ['completion.json', 'consent.json', 'demographics.json', 'start_time.json'],
        'platform_files': {
            'f': 6,  # Facebook: 0, 3, 6, 9, 12, 15
            'i': 6,  # Instagram: 1, 4, 7, 10, 13, 16
            't': 6   # Twitter: 2, 5, 8, 11, 14, 17
        }
    }
    
    # Note: Each platform CSV file also has accompanying _metadata.json and _raw.txt files
    # Total expected files per user: 4 user metadata + 18 CSV + 18 metadata + 18 raw = 58 files
    
    # Sequence to video/session mapping
    SEQUENCE_MAP = {
        # Facebook
        0: (1, 1), 3: (2, 1), 6: (3, 1),
        9: (1, 2), 12: (2, 2), 15: (3, 2),
        # Instagram
        1: (1, 1), 4: (2, 1), 7: (3, 1),
        10: (1, 2), 13: (2, 2), 16: (3, 2),
        # Twitter
        2: (1, 1), 5: (2, 1), 8: (3, 1),
        11: (1, 2), 14: (2, 2), 17: (3, 2)
    }
    
    def __init__(self, source_dir: str, target_dir: str, dry_run: bool = False):
        """
        Initialize the processor.
        
        Args:
            source_dir: Directory containing raw web app data
            target_dir: Root directory for organized output
            dry_run: If True, only simulate operations
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        
        # Set up logging
        self._setup_logging()
        
        # Initialize tracking structures
        self.stats = {
            'total_users': 0,
            'desktop_users': 0,
            'mobile_users': 0,
            'complete_users': 0,
            'incomplete_users': 0,
            'processing_errors': 0,
            'files_processed': 0,
            'txt_files_processed': 0,
            'users_by_device': {'desktop': [], 'mobile': []},
            'broken_users': {'desktop': [], 'mobile': []},
            'errors': []
        }
        
        # User metadata storage
        self.user_metadata: List[UserMetadata] = []
        
    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        log_dir = self.target_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'web_app_processing_{timestamp}.log'
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def extract_user_id(self, filename: str) -> Optional[str]:
        """Extract user ID (hash) from filename."""
        # Pattern variations:
        # - {platform}_{user_hash}_{sequence}.csv
        # - {platform}_{user_hash}_{sequence}_metadata.json
        # - {platform}_{user_hash}_{sequence}_raw.txt
        # - {user_hash}_{type}.json
        
        # Handle platform files (CSV, metadata.json, raw.txt)
        platform_match = re.match(r'^[fit]_([a-f0-9]{32})_\d+(?:\.csv|_metadata\.json|_raw\.txt)$', filename)
        if platform_match:
            return platform_match.group(1)
        
        # Handle user metadata files
        user_match = re.match(r'^([a-f0-9]{32})_\w+\.json$', filename)
        if user_match:
            return user_match.group(1)
        
        return None
    
    def get_user_files(self) -> Dict[str, List[Path]]:
        """Group all files by user ID."""
        user_files = defaultdict(list)
        
        for filepath in self.source_dir.iterdir():
            if filepath.is_file():
                user_id = self.extract_user_id(filepath.name)
                if user_id:
                    user_files[user_id].append(filepath)
                else:
                    self.logger.warning(f"Could not extract user ID from: {filepath.name}")
        
        return dict(user_files)
    
    def validate_user_files(self, user_id: str, files: List[Path]) -> Tuple[bool, List[str]]:
        """
        Validate that user has all required files.
        
        Returns:
            Tuple of (is_complete, missing_files)
        """
        file_names = {f.name for f in files}
        missing = []
        
        # Check metadata files
        for meta_file in self.REQUIRED_FILES['metadata']:
            expected = f"{user_id}_{meta_file}"
            if expected not in file_names:
                missing.append(expected)
        
        # Check platform CSV files
        for platform, count in self.REQUIRED_FILES['platform_files'].items():
            platform_sequences = []
            if platform == 'f':
                platform_sequences = [0, 3, 6, 9, 12, 15]
            elif platform == 'i':
                platform_sequences = [1, 4, 7, 10, 13, 16]
            elif platform == 't':
                platform_sequences = [2, 5, 8, 11, 14, 17]
            
            for seq in platform_sequences:
                expected = f"{platform}_{user_id}_{seq}.csv"
                if expected not in file_names:
                    missing.append(expected)
        
        return len(missing) == 0, missing
    
    def read_demographics(self, user_id: str, files: List[Path]) -> Optional[Dict]:
        """Read and parse demographics JSON for a user."""
        demo_file = next((f for f in files if f.name == f"{user_id}_demographics.json"), None)
        
        if not demo_file:
            return None
        
        try:
            with open(demo_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading demographics for {user_id}: {e}")
            return None
    
    def convert_csv_filename(self, original_name: str, user_hash: str) -> Optional[str]:
        """
        Convert web app CSV filename to TypeNet format.
        
        Example: f_23ad578e675683b69b23d9cf2039bfb8_0.csv -> 1_1_1_23ad578e675683b69b23d9cf2039bfb8.csv
        """
        match = re.match(r'^([fit])_([a-f0-9]{32})_(\d+)\.csv$', original_name)
        if not match:
            return None
        
        platform_letter = match.group(1)
        sequence = int(match.group(3))
        
        # Get platform ID
        platform_id = self.PLATFORM_MAP.get(platform_letter)
        if platform_id is None:
            return None
        
        # Get video and session from sequence
        video_session = self.SEQUENCE_MAP.get(sequence)
        if video_session is None:
            return None
        
        video_id, session_id = video_session
        
        # Create TypeNet format filename using the user hash
        return f"{platform_id}_{video_id}_{session_id}_{user_hash}.csv"
    
    def process_user(self, user_id: str, files: List[Path]) -> bool:
        """
        Process all files for a single user.
        
        Returns:
            True if successfully processed, False otherwise
        """
        self.logger.debug(f"Processing user {user_id}")
        
        # Validate files
        is_complete, missing = self.validate_user_files(user_id, files)
        
        # Read demographics
        demographics = self.read_demographics(user_id, files)
        if not demographics:
            self.logger.error(f"Could not read demographics for {user_id}")
            is_complete = False
        
        # Determine device type
        device_type = 'mobile' if demographics and demographics.get('is_mobile', False) else 'desktop'
        self.stats[f'{device_type}_users'] += 1
        
        # Create user metadata
        if demographics:
            metadata = UserMetadata.from_demographics(user_id, demographics)
            self.user_metadata.append(metadata)
        
        # Determine target directories
        if is_complete:
            user_csv_dir = self.target_dir / device_type / 'raw_data' / user_id
            user_txt_dir = self.target_dir / device_type / 'text' / user_id
            self.stats['complete_users'] += 1
            self.stats['users_by_device'][device_type].append(user_id)
        else:
            user_csv_dir = self.target_dir / device_type / 'broken_data' / user_id
            user_txt_dir = user_csv_dir  # Keep broken data together
            self.stats['incomplete_users'] += 1
            self.stats['broken_users'][device_type].append(user_id)
            self.logger.warning(f"User {user_id} is incomplete. Missing: {missing}")
        
        if not self.dry_run:
            user_csv_dir.mkdir(parents=True, exist_ok=True)
            if is_complete:
                user_txt_dir.mkdir(parents=True, exist_ok=True)
        
        # Process CSV files and their corresponding raw.txt files
        csv_files = [f for f in files if f.name.endswith('.csv') and not f.name.endswith('_metadata.csv')]
        for csv_file in csv_files:
            # Convert CSV filename
            new_name = self.convert_csv_filename(csv_file.name, user_id)
            if new_name:
                # Copy CSV file
                csv_target_path = user_csv_dir / new_name
                try:
                    if not self.dry_run:
                        shutil.copy2(csv_file, csv_target_path)
                    self.stats['files_processed'] += 1
                    self.logger.debug(f"Copied {csv_file.name} -> {new_name}")
                    
                    # Find and copy corresponding raw.txt file
                    raw_txt_name = csv_file.name.replace('.csv', '_raw.txt')
                    raw_txt_file = next((f for f in files if f.name == raw_txt_name), None)
                    
                    if raw_txt_file and is_complete:
                        # Convert raw.txt filename to match CSV naming
                        txt_new_name = new_name.replace('.csv', '.txt')
                        txt_target_path = user_txt_dir / txt_new_name
                        
                        if not self.dry_run:
                            shutil.copy2(raw_txt_file, txt_target_path)
                        self.stats['txt_files_processed'] += 1
                        self.logger.debug(f"Copied {raw_txt_file.name} -> {txt_new_name}")
                    elif not raw_txt_file:
                        self.logger.warning(f"Missing raw.txt file for {csv_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"Error copying {csv_file.name}: {e}")
                    self.stats['processing_errors'] += 1
                    self.stats['errors'].append({
                        'user': user_id,
                        'file': csv_file.name,
                        'error': str(e)
                    })
                    return False
            else:
                self.logger.warning(f"Could not convert filename: {csv_file.name}")
        
        # Copy all original files for broken users (for debugging)
        if not is_complete and not self.dry_run:
            for file in files:
                target_path = user_csv_dir / file.name
                shutil.copy2(file, target_path)
        
        return is_complete
    
    def save_metadata_csv(self):
        """Save user metadata to CSV files."""
        if self.dry_run:
            return
        
        # Separate metadata by device type
        desktop_metadata = [m for m in self.user_metadata if m.device_type == 'desktop']
        mobile_metadata = [m for m in self.user_metadata if m.device_type == 'mobile']
        
        # Save desktop metadata
        if desktop_metadata:
            desktop_meta_dir = self.target_dir / 'desktop' / 'metadata'
            desktop_meta_dir.mkdir(parents=True, exist_ok=True)
            self._write_metadata_csv(desktop_metadata, desktop_meta_dir / 'metadata.csv')
        
        # Save mobile metadata
        if mobile_metadata:
            mobile_meta_dir = self.target_dir / 'mobile' / 'metadata'
            mobile_meta_dir.mkdir(parents=True, exist_ok=True)
            self._write_metadata_csv(mobile_metadata, mobile_meta_dir / 'metadata.csv')
    
    def _write_metadata_csv(self, metadata_list: List[UserMetadata], filepath: Path):
        """Write metadata to CSV file."""
        if not metadata_list:
            return
        
        fieldnames = list(asdict(metadata_list[0]).keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metadata in metadata_list:
                writer.writerow(asdict(metadata))
        
        self.logger.info(f"Saved metadata for {len(metadata_list)} users to {filepath}")
    
    def save_processing_report(self):
        """Save detailed processing report."""
        if self.dry_run:
            return
        
        # Save complete/broken user lists
        for device_type in ['desktop', 'mobile']:
            meta_dir = self.target_dir / device_type / 'metadata'
            meta_dir.mkdir(parents=True, exist_ok=True)
            
            # Save complete users list
            complete_file = meta_dir / 'complete_users.txt'
            with open(complete_file, 'w') as f:
                for user_id in sorted(self.stats['users_by_device'][device_type]):
                    f.write(f"{user_id}\n")
            
            # Save broken users list
            broken_file = meta_dir / 'broken_users.txt'
            with open(broken_file, 'w') as f:
                for user_id in sorted(self.stats['broken_users'][device_type]):
                    f.write(f"{user_id}\n")
        
        # Save detailed processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(self.source_dir),
            'target_directory': str(self.target_dir),
            'dry_run': self.dry_run,
            'statistics': self.stats,
            'platform_mapping': self.PLATFORM_MAP,
            'sequence_mapping': self.SEQUENCE_MAP
        }
        
        summary_file = self.target_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Processing summary saved to {summary_file}")
    
    def process_all_users(self):
        """Process all users in the source directory."""
        self.logger.info(f"Starting web app data processing")
        self.logger.info(f"Source: {self.source_dir}")
        self.logger.info(f"Target: {self.target_dir}")
        self.logger.info(f"Dry run: {self.dry_run}")
        
        # Get all user files
        user_files = self.get_user_files()
        self.stats['total_users'] = len(user_files)
        
        self.logger.info(f"Found {len(user_files)} users to process")
        
        # Create target directory structure
        if not self.dry_run:
            for device_type in ['desktop', 'mobile']:
                # Create main directories
                (self.target_dir / device_type / 'raw_data').mkdir(parents=True, exist_ok=True)
                (self.target_dir / device_type / 'text').mkdir(parents=True, exist_ok=True)
                (self.target_dir / device_type / 'broken_data').mkdir(parents=True, exist_ok=True)
                (self.target_dir / device_type / 'metadata').mkdir(parents=True, exist_ok=True)
        
        # Process each user
        for user_id, files in sorted(user_files.items()):
            try:
                self.process_user(user_id, files)
            except Exception as e:
                self.logger.error(f"Error processing user {user_id}: {e}")
                self.stats['processing_errors'] += 1
                self.stats['errors'].append({
                    'user': user_id,
                    'error': str(e),
                    'type': 'processing'
                })
        
        # Save metadata and reports
        self.save_metadata_csv()
        self.save_processing_report()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary."""
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total users: {self.stats['total_users']}")
        self.logger.info(f"Desktop users: {self.stats['desktop_users']}")
        self.logger.info(f"Mobile users: {self.stats['mobile_users']}")
        self.logger.info(f"Complete users: {self.stats['complete_users']}")
        self.logger.info(f"Incomplete users: {self.stats['incomplete_users']}")
        self.logger.info(f"CSV files processed: {self.stats['files_processed']}")
        self.logger.info(f"Text files processed: {self.stats['txt_files_processed']}")
        self.logger.info(f"Processing errors: {self.stats['processing_errors']}")
        
        if self.stats['errors']:
            self.logger.info("\nFirst 5 errors:")
            for error in self.stats['errors'][:5]:
                self.logger.info(f"  - {error}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process web app keystroke data into TypeNet-compatible format"
    )
    parser.add_argument(
        'source_dir',
        help='Directory containing raw web app data files'
    )
    parser.add_argument(
        'target_dir',
        help='Target directory for organized output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate processing without creating files'
    )
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = WebAppDataProcessor(args.source_dir, args.target_dir, args.dry_run)
    
    try:
        processor.process_all_users()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()