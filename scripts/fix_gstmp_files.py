#!/usr/bin/env python3
"""
Fix .gstmp files by attempting to recover valid CSV files
and handle missing metadata gracefully
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GstmpFixer:
    """Fix .gstmp files and recover valid data"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.stats = {
            "total_gstmp_files": 0,
            "csv_gstmp_files": 0,
            "json_gstmp_files": 0,
            "recovered_csv": 0,
            "recovered_json": 0,
            "unrecoverable_csv": 0,
            "unrecoverable_json": 0,
            "missing_metadata_warned": 0,
            "files_already_exist": 0,
        }
        self.issues = []

    def validate_csv(self, file_path: Path) -> bool:
        """Check if a file is a valid CSV"""
        try:
            with open(file_path, encoding="utf-8") as f:
                # Try to read as CSV
                reader = csv.reader(f)
                rows = list(reader)

                # Check if it has the expected header
                if len(rows) > 0:
                    header = rows[0]
                    # Expected header for keystroke CSV files
                    if len(header) == 3 and header == [
                        "Press or Release",
                        "Key",
                        "Time",
                    ]:
                        return True
                    # Also accept if it has data that looks like keystroke data
                    elif len(rows) > 1:
                        # Check first data row
                        first_row = rows[1] if len(rows) > 1 else rows[0]
                        if len(first_row) == 3 and first_row[0] in ["P", "R"]:
                            return True

                # If file has content but doesn't match expected format,
                # still consider it valid if it's parseable CSV
                return len(rows) > 0

        except Exception as e:
            logger.debug(f"CSV validation failed for {file_path}: {e}")
            return False

    def validate_json(self, file_path: Path) -> bool:
        """Check if a file is valid JSON"""
        try:
            with open(file_path, encoding="utf-8") as f:
                json.load(f)
                return True
        except Exception as e:
            logger.debug(f"JSON validation failed for {file_path}: {e}")
            return False

    def fix_gstmp_file(self, gstmp_file: Path) -> Tuple[bool, str]:
        """
        Attempt to fix a single .gstmp file
        Returns: (success, message)
        """
        # Determine the target filename (remove .gstmp or _.gstmp)
        target_name = gstmp_file.name
        if target_name.endswith("_.gstmp"):
            target_name = target_name[:-7]  # Remove '_.gstmp'
        elif target_name.endswith(".gstmp"):
            target_name = target_name[:-6]  # Remove '.gstmp'
        else:
            return False, f"Unexpected .gstmp pattern: {gstmp_file.name}"

        target_path = gstmp_file.parent / target_name

        # Check if target already exists
        if target_path.exists():
            self.stats["files_already_exist"] += 1
            return True, f"Target already exists: {target_name}"

        # Determine file type and validate
        if target_name.endswith(".csv"):
            self.stats["csv_gstmp_files"] += 1
            if self.validate_csv(gstmp_file):
                # Copy the file without .gstmp extension
                shutil.copy2(gstmp_file, target_path)
                self.stats["recovered_csv"] += 1
                return True, f"Recovered CSV: {target_name}"
            else:
                self.stats["unrecoverable_csv"] += 1
                self.issues.append(f"Unrecoverable CSV: {gstmp_file}")
                return False, f"Invalid CSV content in: {gstmp_file.name}"

        elif target_name.endswith(".json"):
            self.stats["json_gstmp_files"] += 1
            if self.validate_json(gstmp_file):
                # Copy the file without .gstmp extension
                shutil.copy2(gstmp_file, target_path)
                self.stats["recovered_json"] += 1
                return True, f"Recovered JSON: {target_name}"
            else:
                self.stats["unrecoverable_json"] += 1
                self.issues.append(f"Unrecoverable JSON: {gstmp_file}")
                return False, f"Invalid JSON content in: {gstmp_file.name}"

        else:
            return False, f"Unknown file type: {target_name}"

    def check_missing_metadata(self, data_dir: Path) -> List[str]:
        """Check for CSV files missing their metadata"""
        missing_metadata = []

        # Find all CSV files
        for csv_file in data_dir.glob("*.csv"):
            # Skip if it's a gstmp file
            if ".gstmp" in csv_file.name:
                continue

            # Check for corresponding metadata file
            base_name = csv_file.stem
            metadata_file = csv_file.parent / f"{base_name}_metadata.json"

            if not metadata_file.exists():
                # Check if there's a .gstmp version
                gstmp_metadata = csv_file.parent / f"{base_name}_metadata.json_.gstmp"
                if gstmp_metadata.exists():
                    missing_metadata.append(f"{csv_file.name} (has .gstmp metadata)")
                else:
                    missing_metadata.append(f"{csv_file.name} (no metadata at all)")
                self.stats["missing_metadata_warned"] += 1

        return missing_metadata

    def process_directory(self, directory: Path) -> Dict:
        """Process all .gstmp files in a directory"""
        results = {
            "directory": str(directory),
            "fixed_files": [],
            "failed_files": [],
            "missing_metadata": [],
        }

        # Find all .gstmp files
        gstmp_files = list(directory.glob("*.gstmp"))
        self.stats["total_gstmp_files"] += len(gstmp_files)

        if gstmp_files:
            logger.info(f"Found {len(gstmp_files)} .gstmp files in {directory}")

        for gstmp_file in gstmp_files:
            success, message = self.fix_gstmp_file(gstmp_file)
            if success:
                results["fixed_files"].append(message)
                logger.info(f"  ✓ {message}")
            else:
                results["failed_files"].append(message)
                logger.warning(f"  ✗ {message}")

        # Check for missing metadata
        missing_metadata = self.check_missing_metadata(directory)
        if missing_metadata:
            results["missing_metadata"] = missing_metadata
            for item in missing_metadata:
                logger.warning(f"  ⚠ Missing metadata: {item}")

        return results

    def run(self) -> None:
        """Run the fixer on the entire data directory"""
        if not self.data_dir.exists():
            logger.error(f"Directory not found: {self.data_dir}")
            return

        logger.info(f"Processing directory: {self.data_dir}")
        logger.info("=" * 60)

        # Process the main directory first
        results = self.process_directory(self.data_dir)
        all_results = (
            [results]
            if results["fixed_files"]
            or results["failed_files"]
            or results["missing_metadata"]
            else []
        )

        # Then process subdirectories
        for subdir in self.data_dir.rglob("*"):
            if subdir.is_dir():
                # Check if directory contains any .gstmp files
                if list(subdir.glob("*.gstmp")):
                    results = self.process_directory(subdir)
                    all_results.append(results)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total .gstmp files found: {self.stats['total_gstmp_files']}")
        logger.info(f"  CSV .gstmp files: {self.stats['csv_gstmp_files']}")
        logger.info(f"  JSON .gstmp files: {self.stats['json_gstmp_files']}")
        logger.info("\nRecovered files:")
        logger.info(f"  ✓ Recovered CSV files: {self.stats['recovered_csv']}")
        logger.info(f"  ✓ Recovered JSON files: {self.stats['recovered_json']}")
        logger.info(f"  ✓ Files already existed: {self.stats['files_already_exist']}")
        logger.info("\nFailed recoveries:")
        logger.info(f"  ✗ Unrecoverable CSV files: {self.stats['unrecoverable_csv']}")
        logger.info(f"  ✗ Unrecoverable JSON files: {self.stats['unrecoverable_json']}")
        logger.info("\nWarnings:")
        logger.info(
            f"  ⚠ CSV files missing metadata: {self.stats['missing_metadata_warned']}"
        )

        if self.issues:
            logger.info(f"\nDetailed issues ({len(self.issues)}):")
            for issue in self.issues:
                logger.info(f"  - {issue}")

        # Create a report file
        report_path = self.data_dir.parent / "gstmp_fix_report.txt"
        with open(report_path, "w") as f:
            f.write("GSTMP File Recovery Report\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Directory: {self.data_dir}\n\n")
            f.write("Statistics:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nIssues:\n")
            for issue in self.issues:
                f.write(f"  - {issue}\n")

        logger.info(f"\nReport saved to: {report_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix .gstmp files and recover valid data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="artifacts/2025-08-10_20-08-33_loris-mbp-cable-rcn-com/raw_data/web_app_data",
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Version ID to process (e.g., 2025-08-10_20-08-33_loris-mbp-cable-rcn-com)",
    )

    args = parser.parse_args()

    # Determine data directory
    if args.version:
        data_dir = Path(f"artifacts/{args.version}/raw_data/web_app_data")
    else:
        data_dir = Path(args.data_dir)

    # Run the fixer
    fixer = GstmpFixer(data_dir)
    fixer.run()


if __name__ == "__main__":
    main()
