#!/usr/bin/env python3
"""
Download Data Stage
Downloads web app data from Google Cloud Storage

This stage:
- Downloads all files from the GCS uploads bucket
- Validates GCS authentication
- Tracks download metadata in etl_metadata/download/
- Supports dry-run mode for testing
"""

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


class DownloadDataStage:
    """Stage 1: Download web app data from GCS"""

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

        # Stage-specific config
        self.bucket_name = config.get(
            "BUCKET_NAME", "fake-profile-detection-eda-bucket"
        )
        self.source_prefix = "uploads/"
        # self.source_prefix = "uploads_10August2025/"

    def validate_gcs_auth(self) -> bool:
        """Check if GCS authentication is configured"""
        try:
            # Check if gcloud is authenticated
            result = subprocess.run(
                [
                    "gcloud",
                    "auth",
                    "list",
                    "--filter=status:ACTIVE",
                    "--format=value(account)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                logger.error(
                    "Not authenticated with gcloud. Run: gcloud auth application-default login"
                )
                return False

            logger.info(f"Authenticated as: {result.stdout.strip()}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check gcloud auth: {e}")
            if "Reauthentication" in str(e) or "non-interactive" in str(e):
                logger.error("Your authentication has expired. Please re-authenticate:")
                logger.error("  Run: gcloud auth application-default login")
            return False

    def find_gstmp_files(self, directory: Path) -> List[Path]:
        """Find all .gstmp files in a directory"""
        return list(directory.rglob("*.gstmp"))

    def retry_failed_downloads(
        self, output_dir: Path, max_retries: int = 3
    ) -> Dict[str, Any]:
        """Retry downloading files that failed (have .gstmp files)"""
        retry_stats = {
            "initial_gstmp_files": 0,
            "retries_attempted": 0,
            "files_recovered": 0,
            "final_gstmp_files": 0,
            "failed_files": [],
        }

        for attempt in range(max_retries):
            gstmp_files = self.find_gstmp_files(output_dir)

            if attempt == 0:
                retry_stats["initial_gstmp_files"] = len(gstmp_files)

            if not gstmp_files:
                logger.info("No .gstmp files found - all downloads complete")
                break

            logger.info(
                f"Retry attempt {attempt + 1}/{max_retries}: Found {len(gstmp_files)} .gstmp files"
            )
            retry_stats["retries_attempted"] = attempt + 1

            # Convert .gstmp files back to original filenames for retry
            files_to_retry = []
            for gstmp_file in gstmp_files:
                # Remove .gstmp or _.gstmp suffix to get original filename
                original_name = gstmp_file.name
                if original_name.endswith("_.gstmp"):
                    original_name = original_name[:-7]
                elif original_name.endswith(".gstmp"):
                    original_name = original_name[:-6]

                # Construct GCS path
                gcs_path = f"gs://{self.bucket_name}/uploads/{original_name}"
                local_path = output_dir / original_name
                files_to_retry.append((gcs_path, local_path, gstmp_file))

            # Retry downloading each failed file individually
            recovered_count = 0
            for gcs_path, local_path, gstmp_file in files_to_retry:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would retry download: {gcs_path}")
                    continue

                logger.info(f"Retrying: {gcs_path}")

                # Remove the .gstmp file first
                if gstmp_file.exists():
                    gstmp_file.unlink()

                # Try to download the individual file
                cmd = ["gcloud", "storage", "cp", gcs_path, str(local_path)]

                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=60
                    )

                    if (
                        result.returncode == 0
                        and local_path.exists()
                        and local_path.stat().st_size > 0
                    ):
                        logger.info(f"✅ Successfully recovered: {local_path.name}")
                        recovered_count += 1
                    else:
                        logger.warning(f"❌ Failed to recover: {local_path.name}")
                        if result.stderr:
                            logger.debug(f"Error: {result.stderr[:200]}")
                        retry_stats["failed_files"].append(str(gcs_path))

                except subprocess.TimeoutExpired:
                    logger.warning(f"⏰ Timeout downloading: {gcs_path}")
                    retry_stats["failed_files"].append(str(gcs_path))
                except Exception as e:
                    logger.error(f"💥 Exception downloading {gcs_path}: {e}")
                    retry_stats["failed_files"].append(str(gcs_path))

                # Small delay between retries
                time.sleep(0.5)

            retry_stats["files_recovered"] += recovered_count
            logger.info(f"Recovered {recovered_count} files in attempt {attempt + 1}")

            # If we recovered all files, we're done
            if recovered_count == len(files_to_retry):
                logger.info("🎉 All files successfully recovered!")
                break

            # Wait before next retry attempt
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                logger.info(f"Waiting {wait_time}s before next retry attempt...")
                time.sleep(wait_time)

        # Final count of remaining .gstmp files
        final_gstmp_files = self.find_gstmp_files(output_dir)
        retry_stats["final_gstmp_files"] = len(final_gstmp_files)

        if final_gstmp_files:
            logger.warning(
                f"⚠️  {len(final_gstmp_files)} files still incomplete after {max_retries} retry attempts:"
            )
            for gstmp_file in final_gstmp_files:
                logger.warning(f"  - {gstmp_file.name}")
        else:
            logger.info("✅ All files downloaded successfully!")

        return retry_stats

    def get_gcs_files(self) -> List[str]:
        """List files in GCS bucket"""
        try:
            # Use gcloud storage instead of gsutil for better compatibility
            cmd = [
                "gcloud",
                "storage",
                "ls",
                f"gs://{self.bucket_name}/{self.source_prefix}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            files = [
                f
                for f in result.stdout.strip().split("\n")
                if f and not f.endswith("/")
            ]
            return files

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list GCS files: {e}")
            logger.error(
                f"Error output: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}"
            )
            raise RuntimeError(
                f"Cannot access GCS bucket: {e.stderr if hasattr(e, 'stderr') else str(e)}"
            )

    def download_files(self, output_dir: Path) -> Dict[str, Any]:
        """Download files from GCS to local directory"""
        # Ensure parent directory exists
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Download to a temp "uploads" folder first
        temp_download_dir = output_dir.parent / "uploads"

        download_stats = {
            "files_downloaded": 0,
            "total_size": 0,
            "errors": [],
            "start_time": datetime.now().isoformat(),
        }

        try:
            # Download entire uploads directory (no wildcard)
            cmd = [
                "gcloud",
                "storage",
                "cp",
                "-r",
                "--continue-on-error",  # Continue downloading other files if one fails
                f"gs://{self.bucket_name}/uploads",
                str(output_dir.parent) + "/",
            ]

            logger.info(f"Downloading from: gs://{self.bucket_name}/uploads")
            logger.info(f"Temp destination: {temp_download_dir}")
            logger.info(f"Final destination: {output_dir}")

            if not self.dry_run:
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    error_msg = f"gcloud storage cp had some errors (this is common): {result.stderr[:500]}..."
                    logger.warning(error_msg)
                    download_stats["errors"].append(error_msg)
                    # Don't return early - continue with what was downloaded

                # Move the uploads folder to web_app_data
                if temp_download_dir.exists():
                    # Remove existing output_dir if it exists (might have .gstmp files)
                    if output_dir.exists():
                        shutil.rmtree(output_dir)

                    # Rename uploads to web_app_data
                    temp_download_dir.rename(output_dir)
                    logger.info(f"Moved {temp_download_dir} to {output_dir}")
                else:
                    logger.error(
                        f"Download directory {temp_download_dir} not found after download"
                    )
                    download_stats["errors"].append("Download directory not created")

            # Retry failed downloads if any .gstmp files exist
            if output_dir.exists() and not self.dry_run:
                logger.info("Checking for incomplete downloads...")
                retry_stats = self.retry_failed_downloads(output_dir, max_retries=3)
                download_stats["retry_stats"] = retry_stats

                if retry_stats["initial_gstmp_files"] > 0:
                    logger.info(
                        f"Retry summary: {retry_stats['files_recovered']}/{retry_stats['initial_gstmp_files']} "
                        f"files recovered in {retry_stats['retries_attempted']} attempts"
                    )

                # Fail if any files remain incomplete after retries
                if retry_stats["final_gstmp_files"] > 0:
                    error_msg = f"CRITICAL: {retry_stats['final_gstmp_files']} files remain incomplete after retries"
                    logger.error(error_msg)
                    download_stats["errors"].append(error_msg)
                    # Don't raise exception - let caller decide how to handle

            # Count downloaded files
            if output_dir.exists():
                # Count all files recursively
                files = list(output_dir.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                download_stats["files_downloaded"] = file_count
                download_stats["total_size"] = sum(
                    f.stat().st_size for f in files if f.is_file()
                )

        except Exception as e:
            error_msg = f"Download failed: {e!s}"
            logger.error(error_msg)
            download_stats["errors"].append(error_msg)

        download_stats["end_time"] = datetime.now().isoformat()
        return download_stats

    def run(self) -> Path:
        """Execute the download stage"""
        logger.info(f"Starting Download Data stage for version {self.version_id}")

        # Setup output directory following artifacts structure
        artifacts_dir = (
            Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        )
        output_dir = artifacts_dir / "raw_data" / "web_app_data"
        metadata_dir = artifacts_dir / "etl_metadata" / "download"

        if self.local_only:
            logger.info("Local-only mode: Looking for existing data")

            # Check if a specific version was requested
            if (
                self.version_id
                and Path("artifacts") / self.version_id / "raw_data" / "web_app_data"
                == output_dir
            ):
                # User wants to use their current version's data if it exists
                if output_dir.exists() and any(output_dir.iterdir()):
                    logger.info(
                        f"Using existing data from current version: {self.version_id}"
                    )
                    return output_dir

            # Check for test data first
            test_data_dir = Path("test_data/raw_data")
            if test_data_dir.exists() and any(test_data_dir.iterdir()):
                logger.info(f"Using test data from {test_data_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Copy test data to output directory
                import shutil

                for item in test_data_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, output_dir)

                logger.info(
                    f"Copied {len(list(output_dir.iterdir()))} files from test data"
                )
                return output_dir

            # Interactive version selection
            logger.info("No test data found. Checking for available versions...")

            # Find all local versions with data
            artifacts_path = Path("artifacts")
            local_versions = []
            if artifacts_path.exists():
                for version_dir in sorted(artifacts_path.iterdir(), reverse=True):
                    if version_dir.is_dir():
                        raw_data_path = version_dir / "raw_data" / "web_app_data"
                        if raw_data_path.exists() and any(raw_data_path.iterdir()):
                            file_count = len(list(raw_data_path.glob("*")))
                            local_versions.append(
                                {
                                    "version_id": version_dir.name,
                                    "path": raw_data_path,
                                    "file_count": file_count,
                                }
                            )

            if not local_versions:
                logger.warning("No local versions found with data.")
                print("\nOptions:")
                print("1. Download new data from cloud (run without --local-only)")
                print("2. Place test data in test_data/raw_data/")
                print("3. Exit")

                choice = input("\nSelect option (1-3): ").strip()
                if choice == "1":
                    logger.info(
                        "Please run without --local-only flag to download from cloud"
                    )
                elif choice == "2":
                    logger.info(
                        "Place your test data files in test_data/raw_data/ and run again"
                    )
                sys.exit(0)
            else:
                # Show available versions
                print(f"\nFound {len(local_versions)} local version(s) with data:")
                for i, ver in enumerate(local_versions, 1):
                    print(f"{i}. {ver['version_id']} ({ver['file_count']} files)")

                print(
                    f"\n{len(local_versions) + 1}. Check cloud for available versions"
                )
                print(f"{len(local_versions) + 2}. Exit")

                while True:
                    try:
                        choice = int(
                            input(f"\nSelect version (1-{len(local_versions) + 2}): ")
                        )
                        if 1 <= choice <= len(local_versions):
                            # Use selected local version
                            selected = local_versions[choice - 1]
                            logger.info(
                                f"Using data from version: {selected['version_id']}"
                            )

                            if not self.dry_run:
                                output_dir.mkdir(parents=True, exist_ok=True)
                                import shutil

                                for item in selected["path"].iterdir():
                                    if item.is_file():
                                        shutil.copy2(item, output_dir)
                                logger.info(
                                    f"Copied {len(list(output_dir.iterdir()))} files from {selected['version_id']}"
                                )
                            return output_dir
                        elif choice == len(local_versions) + 1:
                            # List cloud versions
                            logger.info("Checking cloud for available versions...")
                            bucket_name = self.config.get("BUCKET_NAME")
                            if bucket_name:
                                cmd = [
                                    "gcloud",
                                    "storage",
                                    "ls",
                                    f"gs://{bucket_name}/artifacts/",
                                ]
                                result = subprocess.run(
                                    cmd, capture_output=True, text=True
                                )
                                if result.returncode == 0:
                                    print("\nCloud versions:")
                                    print(result.stdout)
                                else:
                                    print("Failed to list cloud versions")
                            print(
                                "\nTo download from cloud, run without --local-only flag"
                            )
                            sys.exit(0)
                        elif choice == len(local_versions) + 2:
                            sys.exit(0)
                        else:
                            print("Invalid choice")
                    except ValueError:
                        print("Please enter a number")
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        sys.exit(0)

        # Validate authentication
        if not self.validate_gcs_auth():
            logger.error("GCS authentication failed")
            logger.error("Please ensure you are authenticated with Google Cloud:")
            logger.error("  1. Run: gcloud auth login")
            logger.error("  2. Set project: gcloud config set project YOUR_PROJECT_ID")
            logger.error("  3. Or use: gcloud auth application-default login")
            raise RuntimeError("GCS authentication failed. See above for instructions.")

        # List available files
        logger.info("Listing files in GCS...")
        gcs_files = self.get_gcs_files()
        logger.info(f"Found {len(gcs_files)} files in GCS")

        if self.dry_run:
            logger.info("DRY RUN: Would download the following files:")
            for f in gcs_files[:10]:  # Show first 10
                logger.info(f"  - {f}")
            if len(gcs_files) > 10:
                logger.info(f"  ... and {len(gcs_files) - 10} more files")
        else:
            # Download files
            download_stats = self.download_files(output_dir)

            # Save download metadata in etl_metadata structure
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Create manifest
            manifest = {
                "version_id": self.version_id,
                "stage": "download",
                "source": f"gs://{self.bucket_name}/{self.source_prefix}",
                "destination": str(output_dir),
                "files_downloaded": download_stats["files_downloaded"],
                "total_size": download_stats["total_size"],
                "timestamp": datetime.now().isoformat(),
            }

            with open(metadata_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Save detailed download log
            with open(metadata_dir / "download_log.json", "w") as f:
                json.dump(download_stats, f, indent=2)

            logger.info(f"Downloaded {download_stats['files_downloaded']} files")
            logger.info(
                f"Total size: {download_stats['total_size'] / 1024 / 1024:.2f} MB"
            )

            # Check for errors
            if download_stats["errors"]:
                error_count = len(download_stats["errors"])
                logger.warning(f"Download completed with {error_count} warning(s)")
                for error in download_stats["errors"]:
                    logger.warning(f"  - {error}")

                # Only fail if we have no files or too many errors
                if download_stats["files_downloaded"] == 0:
                    raise RuntimeError("Download failed: No files downloaded")
                elif error_count > 10:  # More than 10 errors is concerning
                    logger.error(f"Too many download errors: {error_count}")
                    raise RuntimeError(
                        f"Download failed with {error_count} errors. Check logs for details."
                    )
                else:
                    logger.info(
                        f"Download completed with minor issues: {error_count} file(s) had errors but {download_stats['files_downloaded']} files were successfully downloaded"
                    )

            # Check if any files were actually downloaded
            if download_stats["files_downloaded"] == 0:
                logger.error("No files were downloaded from cloud storage")
                raise RuntimeError(
                    "Download failed: No files retrieved from cloud storage"
                )

        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                "download_data",
                {
                    "output_dir": str(output_dir),
                    "files_count": len(list(output_dir.glob("*"))),
                    "completed_at": datetime.now().isoformat(),
                },
            )

        return output_dir


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
) -> Path:
    """Entry point for the pipeline orchestrator"""
    stage = DownloadDataStage(version_id, config, dry_run, local_only)
    return stage.run()


if __name__ == "__main__":
    # For testing the stage independently
    import click

    from scripts.utils.config_manager import get_config

    @click.command()
    @click.option("--version-id", help="Version ID to use")
    @click.option("--dry-run", is_flag=True, help="Preview without downloading")
    @click.option("--local-only", is_flag=True, help="Skip download, use existing data")
    def main(version_id, dry_run, local_only):
        """Test Stage 1: Download Data independently"""
        logging.basicConfig(level=logging.INFO)

        config = get_config()
        vm = VersionManager()

        if not version_id:
            version_id = vm.create_version_id()
            logger.info(f"Created version ID: {version_id}")

        output_dir = run(version_id, config._config, dry_run, local_only)
        logger.info(f"Stage complete. Output: {output_dir}")

    main()
