#!/usr/bin/env python3
"""
Clean Data Stage
Maps and organizes web app files into user-centric directory structure

This stage:
- Takes flat web app files and organizes them by user
- Validates data completeness for each user
- Separates complete vs broken/incomplete data
- Extracts and stores metadata about users and data quality
- Saves cleaning artifacts in etl_metadata/cleaning/
- Creates a structure suitable for downstream processing
"""

import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


@dataclass
class UserMetadata:
    """Stores user demographic and device information"""

    user_id: str
    # Demographics fields
    email: str = ""
    gender: str = ""
    age: str = ""  # Changed from age_range for compatibility
    handedness: str = ""
    education: str = ""
    consent_email_for_future: bool = False
    submission_timestamp: str = ""
    
    # Device info fields
    is_mobile: bool = False
    device_type: str = "unknown"
    user_agent: str = ""
    screen_width: int = 0
    screen_height: int = 0
    window_width: int = 0
    window_height: int = 0
    touch_capable: bool = False
    platform: str = ""
    vendor: str = ""
    
    # Consent fields
    consented: str = ""
    consent_timestamp: str = ""
    consent_user_agent: str = ""
    consent_url: str = ""
    
    # Data quality fields
    has_timestamp_files: bool = False  # Indicates broken data

    @classmethod
    def from_demographics(cls, user_id: str, demo_data: Dict) -> "UserMetadata":
        """Create UserMetadata from demographics JSON"""
        device_info = demo_data.get("device_info", {})
        return cls(
            user_id=user_id,
            email=demo_data.get("email", ""),
            gender=demo_data.get("gender", ""),
            age=demo_data.get("age", demo_data.get("age_range", "")),  # Handle both fields
            handedness=demo_data.get("handedness", ""),
            education=demo_data.get("education", ""),
            consent_email_for_future=demo_data.get("consent_email_for_future", False),
            submission_timestamp=demo_data.get("submission_timestamp", ""),
            is_mobile=demo_data.get("is_mobile", device_info.get("isMobile", False)),
            device_type=demo_data.get("device_type", device_info.get("deviceType", "unknown")),
            user_agent=device_info.get("userAgent", ""),
            screen_width=device_info.get("screenWidth", 0),
            screen_height=device_info.get("screenHeight", 0),
            window_width=device_info.get("windowWidth", 0),
            window_height=device_info.get("windowHeight", 0),
            touch_capable=device_info.get("touchCapable", False),
            platform=device_info.get("platform", ""),
            vendor=device_info.get("vendor", ""),
        )


class CleanDataStage:
    """Stage 2: Map and organize web app files"""

    # Platform mapping
    PLATFORM_MAP = {
        "f": 1,  # Facebook
        "i": 2,  # Instagram
        "t": 3,  # Twitter
    }

    # Expected files per user
    REQUIRED_METADATA_FILES = ["consent.json", "demographics.json", "start_time.json"]
    OPTIONAL_METADATA_FILES = ["completion.json"]

    # Expected keystroke files per platform
    PLATFORM_SEQUENCES = {
        "facebook": [0, 3, 6, 9, 12, 15],  # Facebook
        "instagram": [1, 4, 7, 10, 13, 16],  # Instagram
        "twitter": [2, 5, 8, 11, 14, 17],  # Twitter
    }

    # Sequence to video/session mapping
    SEQUENCE_MAP = {
        # Facebook
        0: (1, 1),
        3: (2, 1),
        6: (3, 1),
        9: (1, 2),
        12: (2, 2),
        15: (3, 2),
        # Instagram
        1: (1, 1),
        4: (2, 1),
        7: (3, 1),
        10: (1, 2),
        13: (2, 2),
        16: (3, 2),
        # Twitter
        2: (1, 1),
        5: (2, 1),
        8: (3, 1),
        11: (1, 2),
        14: (2, 2),
        17: (3, 2),
    }

    def __init__(
        self,
        version_id: str,
        config: Dict[str, Any],
        dry_run: bool = False,
        local_only: bool = False,
        version_manager: Optional[VersionManager] = None,
    ):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.version_manager = version_manager or VersionManager()

        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "total_users": 0,
            "complete_users": 0,
            "broken_users": 0,
            "desktop_users": 0,
            "mobile_users": 0,
            "processing_errors": [],
            "data_issues": [],
        }

    def extract_user_id(self, filename: str) -> Optional[str]:
        """Extract user ID (hash) from filename"""
        # Match patterns like:
        # - f_3741e927ab7d45a7ca19ed47a3eb5864_0.csv (platform files)
        # - 3741e927ab7d45a7ca19ed47a3eb5864_consent.json (metadata files)
        # - 1_1_1_3741e927ab7d45a7ca19ed47a3eb5864.csv (TypeNet format)
        # - 06967d28_4d00f793fa2a988dfa53eef2a9155837_start_time.json (short prefix format)
        # - 2025-07-06T19-29-57-241Z_3b1da76c_bf369b54edeaf668131aff2c16eee9f9_consent.json (timestamp format)

        # Try platform file pattern
        match = re.search(r"[fit]_([a-f0-9]{32})_\d+", filename)
        if match:
            return match.group(1)

        # Try metadata file pattern (standard)
        match = re.search(r"^([a-f0-9]{32})_", filename)
        if match:
            return match.group(1)

        # Try TypeNet format pattern (Platform_Session_Video_UserID.csv)
        match = re.search(r"^\d+_\d+_\d+_([a-f0-9]{32})", filename)
        if match:
            return match.group(1)

        # Check for short prefix format (broken files) - 8 char prefix
        match = re.search(r"^[a-f0-9]{8}_([a-f0-9]{32})_", filename)
        if match:
            return match.group(1)

        # Check for timestamp format (broken files) - still extract user ID
        match = re.search(r"[0-9T\-Z]+_[a-f0-9]+_([a-f0-9]{32})_", filename)
        if match:
            return match.group(1)

        return None
    
    def has_timestamp_files(self, files: List[Path]) -> bool:
        """Check if any files have timestamp prefix or short prefix (indicating broken data)"""
        for file in files:
            # Check for timestamp format
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z_", file.name):
                return True
            # Check for short prefix format (8 hex chars)
            if re.match(r"^[a-f0-9]{8}_[a-f0-9]{32}_", file.name):
                return True
        return False

    def group_files_by_user(self, input_dir: Path) -> Dict[str, List[Path]]:
        """Group all files by user ID"""
        user_files = defaultdict(list)

        for filepath in input_dir.iterdir():
            if filepath.is_file():
                user_id = self.extract_user_id(filepath.name)
                if user_id:
                    user_files[user_id].append(filepath)
                else:
                    logger.warning(f"Could not extract user ID from: {filepath.name}")
                    self.stats["data_issues"].append(f"Unknown file: {filepath.name}")

        return dict(user_files)

    def validate_user_files(
        self, user_id: str, files: List[Path]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that user has required files

        Returns:
            Tuple of (is_complete, missing_required, missing_optional)
        """
        file_names = {f.name for f in files}
        missing_required = []
        missing_optional = []

        # First check if user has timestamp files (broken data)
        if self.has_timestamp_files(files):
            missing_required.append("User has timestamp-prefixed files (broken data from web app error)")
            return False, missing_required, missing_optional

        # Check required metadata files
        for meta_file in self.REQUIRED_METADATA_FILES:
            expected = f"{user_id}_{meta_file}"
            if expected not in file_names:
                missing_required.append(expected)

        # Check optional metadata files
        for meta_file in self.OPTIONAL_METADATA_FILES:
            expected = f"{user_id}_{meta_file}"
            if expected not in file_names:
                missing_optional.append(expected)

        # Check keystroke files - handle new format with f_, i_, t_ prefixes
        complete_platforms = 0
        platform_completeness = {}

        for platform, sequences in self.PLATFORM_SEQUENCES.items():
            platform_files = []
            prefix = platform[0]  # 'f' for facebook, 'i' for instagram, 't' for twitter

            for seq in sequences:
                csv_file = f"{prefix}_{user_id}_{seq}.csv"

                if csv_file in file_names:
                    platform_files.append(seq)

            platform_completeness[platform] = {
                "expected": len(sequences),
                "found": len(platform_files),
                "sequences": platform_files,
            }

            if len(platform_files) == len(sequences):
                complete_platforms += 1

        # User is complete if they have all required metadata and ALL platforms complete (18 files)
        has_all_platforms = complete_platforms == len(self.PLATFORM_SEQUENCES)
        is_complete = len(missing_required) == 0 and has_all_platforms

        # Add platform info to missing if not all platforms complete
        if not has_all_platforms:
            missing_required.append(
                f"Incomplete platform data (need all 18 files): {platform_completeness}"
            )

        return is_complete, missing_required, missing_optional

    def read_demographics(self, user_id: str, files: List[Path]) -> Optional[Dict]:
        """Read and parse demographics JSON for a user"""
        # Try old format first
        demo_file = next(
            (f for f in files if f.name == f"{user_id}_demographics.json"), None
        )

        # Try new format if old format not found
        if not demo_file:
            demo_file = next(
                (f for f in files if f"{user_id}_demographics.json" in f.name), None
            )

        if not demo_file:
            return None

        try:
            with open(demo_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading demographics for {user_id}: {e}")
            self.stats["processing_errors"].append(
                f"Demographics read error for {user_id}: {e!s}"
            )
            return None

    def read_consent(self, user_id: str, files: List[Path]) -> Optional[Dict]:
        """Read and parse consent JSON for a user"""
        # Try old format first
        consent_file = next(
            (f for f in files if f.name == f"{user_id}_consent.json"), None
        )

        # Try new format if old format not found
        if not consent_file:
            consent_file = next(
                (f for f in files if f"{user_id}_consent.json" in f.name), None
            )

        if not consent_file:
            return None

        try:
            with open(consent_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading consent for {user_id}: {e}")
            return None

    def convert_csv_filename(self, original_name: str, user_id: str) -> Optional[str]:
        """
        Convert web app CSV filename to TypeNet format
        Example: f_3741e927ab7d45a7ca19ed47a3eb5864_0.csv -> 1_1_1_3741e927ab7d45a7ca19ed47a3eb5864.csv
        """
        match = re.match(r"^([fit])_([a-f0-9]{32})_(\d+)\.csv$", original_name)
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

        # Create TypeNet format filename
        return f"{platform_id}_{video_id}_{session_id}_{user_id}.csv"

    def process_user(self, user_id: str, files: List[Path], output_base: Path) -> bool:
        """Process all files for a single user"""
        # Validate files
        is_complete, missing_required, missing_optional = self.validate_user_files(
            user_id, files
        )

        # Read demographics to determine device type
        demographics = self.read_demographics(user_id, files)

        # Default to desktop if demographics unavailable
        device_type = "desktop"
        if demographics:
            is_mobile = demographics.get("is_mobile", False)
            device_type = "mobile" if is_mobile else "desktop"

            # Update stats
            if device_type == "mobile":
                self.stats["mobile_users"] += 1
            else:
                self.stats["desktop_users"] += 1

        # Determine output directories
        if is_complete:
            csv_dir = output_base / device_type / "raw_data" / user_id
            text_dir = output_base / device_type / "text" / user_id
            self.stats["complete_users"] += 1
        else:
            csv_dir = output_base / device_type / "broken_data" / user_id
            # Broken users also get text directory under broken_data
            text_dir = output_base / device_type / "broken_data" / "text" / user_id
            self.stats["broken_users"] += 1

            # Log missing files at debug level - this is expected for incomplete users
            logger.debug(f"User {user_id} has incomplete data:")
            if missing_required:
                logger.debug(f"  Missing required: {missing_required}")
            if missing_optional:
                logger.debug(f"  Missing optional: {missing_optional}")

        # Create directories
        if not self.dry_run:
            csv_dir.mkdir(parents=True, exist_ok=True)
            if text_dir:
                text_dir.mkdir(parents=True, exist_ok=True)

        # Process files
        files_copied = 0
        for file_path in files:
            filename = file_path.name

            # Skip JSON metadata files - they don't go in user directories
            if filename.endswith(".json"):
                # These files (consent, demographics, etc.) are not copied to user dirs
                continue

            # Process text files - move to text directory
            elif filename.endswith("_raw.txt"):
                if text_dir and not self.dry_run:
                    # Convert filename format for text files
                    # From: f_3741e927ab7d45a7ca19ed47a3eb5864_0_raw.txt
                    # To: 1_1_1_3741e927ab7d45a7ca19ed47a3eb5864.txt
                    match = re.match(
                        r"^([fit])_([a-f0-9]{32})_(\d+)_raw\.txt$", filename
                    )
                    if match:
                        platform_letter = match.group(1)
                        user_id_from_file = match.group(2)
                        sequence = int(match.group(3))

                        platform_id = self.PLATFORM_MAP.get(platform_letter)
                        video_session = self.SEQUENCE_MAP.get(sequence)

                        if platform_id and video_session:
                            video_id, session_id = video_session
                            new_name = f"{platform_id}_{video_id}_{session_id}_{user_id_from_file}.txt"
                            dest_path = text_dir / new_name
                            shutil.copy2(file_path, dest_path)
                            files_copied += 1
                    else:
                        # If can't convert, skip this file
                        logger.warning(f"Could not convert text filename: {filename}")

            # Skip metadata JSON files (*_metadata.json)
            elif filename.endswith("_metadata.json"):
                continue

            # Convert and copy CSV files
            elif filename.endswith(".csv"):
                # Try to convert to TypeNet format
                new_name = self.convert_csv_filename(filename, user_id)
                if new_name:
                    dest_path = csv_dir / new_name
                else:
                    # Keep original name if conversion fails
                    dest_path = csv_dir / filename

                if not self.dry_run:
                    shutil.copy2(file_path, dest_path)
                files_copied += 1

        logger.debug(f"Processed user {user_id}: {files_copied} files")
        return is_complete

    def generate_metadata_files(
        self, output_base: Path, user_metadata: Dict[str, UserMetadata]
    ):
        """Generate metadata CSV files for complete and broken users"""
        for device_type in ["desktop", "mobile"]:
            device_dir = output_base / device_type
            
            # Create metadata directories for both raw_data and broken_data
            raw_metadata_dir = device_dir / "metadata"
            broken_metadata_dir = device_dir / "broken_data" / "metadata"

            # Always create metadata directories
            if not self.dry_run:
                raw_metadata_dir.mkdir(parents=True, exist_ok=True)
                broken_metadata_dir.mkdir(parents=True, exist_ok=True)

            # Lists of users
            complete_users = []
            broken_users = []

            # Collect user lists
            raw_data_dir = device_dir / "raw_data"
            if raw_data_dir.exists():
                complete_users = [
                    d.name for d in raw_data_dir.iterdir() 
                    if d.is_dir() and re.match(r'^[a-f0-9]{32}$', d.name)
                ]

            broken_data_dir = device_dir / "broken_data"
            if broken_data_dir.exists():
                broken_users = [
                    d.name for d in broken_data_dir.iterdir() 
                    if d.is_dir() and re.match(r'^[a-f0-9]{32}$', d.name)
                ]

            # Write user lists
            if not self.dry_run:
                with open(raw_metadata_dir / "complete_users.txt", "w") as f:
                    f.write("\n".join(sorted(complete_users)))

                with open(raw_metadata_dir / "broken_users.txt", "w") as f:
                    f.write("\n".join(sorted(broken_users)))

                # Define all metadata fields
                csv_headers = [
                    "user_id", "status", 
                    # Demographics
                    "email", "gender", "age", "handedness", "education", 
                    "consent_email_for_future", "submission_timestamp",
                    # Device info
                    "is_mobile", "device_type", "user_agent", 
                    "screen_width", "screen_height", "window_width", "window_height",
                    "touch_capable", "platform", "vendor",
                    # Consent info
                    "consented", "consent_timestamp", "consent_user_agent", "consent_url",
                    # Data quality
                    "has_timestamp_files"
                ]

                # Write main metadata CSV
                with open(raw_metadata_dir / "metadata.csv", "w") as f:
                    f.write(",".join(csv_headers) + "\n")

                    # Write user data
                    for user_id in complete_users + broken_users:
                        status = "complete" if user_id in complete_users else "broken"
                        meta = user_metadata.get(user_id)

                        if meta:
                            row = [
                                user_id, status,
                                meta.email, meta.gender, meta.age, meta.handedness, meta.education,
                                str(meta.consent_email_for_future), meta.submission_timestamp,
                                str(meta.is_mobile), meta.device_type, meta.user_agent,
                                str(meta.screen_width), str(meta.screen_height), 
                                str(meta.window_width), str(meta.window_height),
                                str(meta.touch_capable), meta.platform, meta.vendor,
                                meta.consented, meta.consent_timestamp, 
                                meta.consent_user_agent, meta.consent_url,
                                str(meta.has_timestamp_files)
                            ]
                            # Escape commas in fields
                            row = [f'"{field}"' if "," in str(field) else str(field) for field in row]
                            f.write(",".join(row) + "\n")
                        else:
                            # Write empty row with just user_id and status
                            empty_row = [user_id, status] + [""] * (len(csv_headers) - 2)
                            f.write(",".join(empty_row) + "\n")

                # Write broken users metadata CSV
                if broken_users:
                    with open(broken_metadata_dir / "metadata.csv", "w") as f:
                        f.write(",".join(csv_headers) + "\n")

                        for user_id in broken_users:
                            meta = user_metadata.get(user_id)
                            if meta:
                                row = [
                                    user_id, "broken",
                                    meta.email, meta.gender, meta.age, meta.handedness, meta.education,
                                    str(meta.consent_email_for_future), meta.submission_timestamp,
                                    str(meta.is_mobile), meta.device_type, meta.user_agent,
                                    str(meta.screen_width), str(meta.screen_height),
                                    str(meta.window_width), str(meta.window_height),
                                    str(meta.touch_capable), meta.platform, meta.vendor,
                                    meta.consented, meta.consent_timestamp,
                                    meta.consent_user_agent, meta.consent_url,
                                    str(meta.has_timestamp_files)
                                ]
                                # Escape commas in fields
                                row = [f'"{field}"' if "," in str(field) else str(field) for field in row]
                                f.write(",".join(row) + "\n")
                            else:
                                empty_row = [user_id, "broken"] + [""] * (len(csv_headers) - 2)
                                f.write(",".join(empty_row) + "\n")

    def run(self, input_dir: Path) -> Path:
        """Execute the clean data stage"""
        logger.info(f"Starting Clean Data stage for version {self.version_id}")
        logger.info(f"Input directory: {input_dir}")

        # Setup output directories following artifacts structure
        artifacts_dir = (
            Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        )
        output_base = artifacts_dir / "cleaned_data"
        metadata_dir = artifacts_dir / "etl_metadata" / "cleaning"

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Count input files
        input_files = list(input_dir.glob("*"))
        self.stats["total_files"] = len(input_files)
        logger.info(f"Found {self.stats['total_files']} files to process")

        # Create base directory structure (even if no users found)
        if not self.dry_run:
            for device_type in ["desktop", "mobile"]:
                # Create raw_data structure
                for subdir in ["raw_data", "metadata", "text"]:
                    dir_path = output_base / device_type / subdir
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create broken_data structure with its own subdirectories
                broken_base = output_base / device_type / "broken_data"
                broken_base.mkdir(parents=True, exist_ok=True)
                (broken_base / "metadata").mkdir(parents=True, exist_ok=True)
                (broken_base / "text").mkdir(parents=True, exist_ok=True)

        # Group files by user
        user_files = self.group_files_by_user(input_dir)
        self.stats["total_users"] = len(user_files)
        logger.info(f"Found {self.stats['total_users']} unique users")

        # Process each user
        user_metadata = {}
        for user_id, files in user_files.items():
            try:
                # Read demographics and consent data
                demographics = self.read_demographics(user_id, files)
                consent_data = self.read_consent(user_id, files)

                # Create metadata object
                if demographics:
                    user_metadata[user_id] = UserMetadata.from_demographics(
                        user_id, demographics
                    )
                else:
                    # Create minimal metadata with just user_id
                    user_metadata[user_id] = UserMetadata(user_id=user_id)

                # Add consent data if available
                if consent_data:
                    user_metadata[user_id].consented = consent_data.get("consented", "")
                    user_metadata[user_id].consent_timestamp = consent_data.get("timestamp", "")
                    user_metadata[user_id].consent_user_agent = consent_data.get("userAgent", "")
                    user_metadata[user_id].consent_url = consent_data.get("url", "")

                # Check if user has timestamp files (broken data)
                user_metadata[user_id].has_timestamp_files = self.has_timestamp_files(files)

                # Process user files
                self.process_user(user_id, files, output_base)

            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                self.stats["processing_errors"].append(f"User {user_id}: {e!s}")

        # Generate metadata files
        self.generate_metadata_files(output_base, user_metadata)

        # Save cleaning artifacts in etl_metadata
        if not self.dry_run:
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Cleaning report
            cleaning_report = {
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(input_dir),
                "output_dir": str(output_base),
                "summary": {
                    "total_files": self.stats["total_files"],
                    "total_users": self.stats["total_users"],
                    "complete_users": self.stats["complete_users"],
                    "broken_users": self.stats["broken_users"],
                    "desktop_users": self.stats["desktop_users"],
                    "mobile_users": self.stats["mobile_users"],
                },
            }

            with open(metadata_dir / "cleaning_report.json", "w") as f:
                json.dump(cleaning_report, f, indent=2)

            # Cleaning stats (detailed)
            with open(metadata_dir / "cleaning_stats.json", "w") as f:
                json.dump(self.stats, f, indent=2)

            # Validation errors
            if self.stats["data_issues"] or self.stats["processing_errors"]:
                validation_errors = {
                    "data_issues": self.stats["data_issues"],
                    "processing_errors": self.stats["processing_errors"],
                    "timestamp": datetime.now().isoformat(),
                }
                with open(metadata_dir / "validation_errors.json", "w") as f:
                    json.dump(validation_errors, f, indent=2)

        # Log summary
        logger.info("Processing complete:")
        logger.info(f"  Total users: {self.stats['total_users']}")
        if self.stats["total_users"] > 0:
            complete_pct = (
                self.stats["complete_users"] / self.stats["total_users"] * 100
            )
            broken_pct = self.stats["broken_users"] / self.stats["total_users"] * 100
            logger.info(
                f"  Complete users: {self.stats['complete_users']} ({complete_pct:.1f}%)"
            )
            logger.info(
                f"  Incomplete users: {self.stats['broken_users']} ({broken_pct:.1f}%)"
            )
        else:
            logger.info(f"  Complete users: {self.stats['complete_users']}")
            logger.info(f"  Incomplete users: {self.stats['broken_users']}")
        logger.info(f"  Desktop users: {self.stats['desktop_users']}")
        logger.info(f"  Mobile users: {self.stats['mobile_users']}")

        if self.stats["processing_errors"]:
            logger.warning(
                f"  Processing errors: {len(self.stats['processing_errors'])}"
            )

        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                "clean_data",
                {
                    "output_dir": str(output_base),
                    "stats": self.stats,
                    "completed_at": datetime.now().isoformat(),
                },
            )

        return output_base


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
) -> Path:
    """Entry point for the pipeline orchestrator"""
    # Get input directory from previous stage
    vm = VersionManager()
    version_info = vm.get_version(version_id)

    if not version_info or "download_data" not in version_info.get("stages", {}):
        # Default input directory following new structure
        artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts")) / version_id
        input_dir = artifacts_dir / "raw_data" / "web_app_data"
    else:
        # Handle both output_dir and output_path for compatibility
        download_info = version_info["stages"]["download_data"]
        path_key = "output_path" if "output_path" in download_info else "output_dir"
        input_dir = Path(download_info[path_key])

    stage = CleanDataStage(version_id, config, dry_run, local_only)
    return stage.run(input_dir)


if __name__ == "__main__":
    # For testing the stage independently
    import click

    from scripts.utils.config_manager import get_config

    @click.command()
    @click.option("--version-id", help="Version ID to use")
    @click.option("--input-dir", help="Input directory (overrides default)")
    @click.option("--dry-run", is_flag=True, help="Preview without processing")
    def main(version_id, input_dir, dry_run):
        """Test Stage 2: Clean Data independently"""
        logging.basicConfig(level=logging.INFO)

        config = get_config().config
        vm = VersionManager()

        if not version_id:
            version_id = vm.create_version_id()
            logger.info(f"Created version ID: {version_id}")

        if input_dir:
            # Override the default input directory
            stage = CleanDataStage(version_id, config, dry_run)
            output_dir = stage.run(Path(input_dir))
        else:
            output_dir = run(version_id, config, dry_run)

        logger.info(f"Stage complete. Output: {output_dir}")

    main()
