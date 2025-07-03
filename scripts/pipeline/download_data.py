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

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import EnhancedVersionManager as VersionManager
from scripts.utils.cloud_artifact_manager import CloudArtifactManager

logger = logging.getLogger(__name__)


class DownloadDataStage:
    """Stage 1: Download web app data from GCS"""
    
    def __init__(self, version_id: str, config: Dict[str, Any], 
                 dry_run: bool = False, local_only: bool = False,
                 version_manager: Optional[VersionManager] = None):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.version_manager = version_manager or VersionManager()
        
        # Stage-specific config
        self.bucket_name = config.get("BUCKET_NAME", "fake-profile-detection-eda-bucket")
        self.source_prefix = "uploads/"
        
    def validate_gcs_auth(self) -> bool:
        """Check if GCS authentication is configured"""
        try:
            # Check if gcloud is authenticated
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True, text=True, check=True
            )
            
            if not result.stdout.strip():
                logger.error("Not authenticated with gcloud. Run: gcloud auth application-default login")
                return False
                
            logger.info(f"Authenticated as: {result.stdout.strip()}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check gcloud auth: {e}")
            if "Reauthentication" in str(e) or "non-interactive" in str(e):
                logger.error("Your authentication has expired. Please re-authenticate:")
                logger.error("  Run: gcloud auth application-default login")
            return False
            
    def get_gcs_files(self) -> List[str]:
        """List files in GCS bucket"""
        try:
            # Use gcloud storage instead of gsutil for better compatibility
            cmd = ["gcloud", "storage", "ls", f"gs://{self.bucket_name}/{self.source_prefix}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            files = [f for f in result.stdout.strip().split('\n') if f and not f.endswith('/')]
            return files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list GCS files: {e}")
            logger.error(f"Error output: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            raise RuntimeError(f"Cannot access GCS bucket: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            
    def download_files(self, output_dir: Path) -> Dict[str, Any]:
        """Download files from GCS to local directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        download_stats = {
            "files_downloaded": 0,
            "total_size": 0,
            "errors": [],
            "start_time": datetime.now().isoformat(),
        }
        
        try:
            # Use gcloud storage cp for better compatibility
            cmd = [
                "gcloud", "storage", "cp", "-r",
                f"gs://{self.bucket_name}/{self.source_prefix}*",
                str(output_dir) + "/"
            ]
            
            logger.info(f"Downloading from: gs://{self.bucket_name}/{self.source_prefix}")
            logger.info(f"Destination: {output_dir}")
            
            if not self.dry_run:
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    error_msg = f"gcloud storage cp failed: {result.stderr}"
                    logger.error(error_msg)
                    download_stats["errors"].append(error_msg)
                    return download_stats
                    
            # Count downloaded files
            if output_dir.exists():
                files = list(output_dir.glob("*"))
                download_stats["files_downloaded"] = len(files)
                download_stats["total_size"] = sum(f.stat().st_size for f in files if f.is_file())
                
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            download_stats["errors"].append(error_msg)
            
        download_stats["end_time"] = datetime.now().isoformat()
        return download_stats
        
    def run(self) -> Path:
        """Execute the download stage"""
        logger.info(f"Starting Download Data stage for version {self.version_id}")
        
        # Setup output directory following artifacts structure
        artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        output_dir = artifacts_dir / "raw_data" / "web_app_data"
        metadata_dir = artifacts_dir / "etl_metadata" / "download"
        
        if self.local_only:
            logger.info("Local-only mode: Looking for existing data")
            
            # Check if a specific version was requested
            if self.version_id and Path("artifacts") / self.version_id / "raw_data" / "web_app_data" == output_dir:
                # User wants to use their current version's data if it exists
                if output_dir.exists() and any(output_dir.iterdir()):
                    logger.info(f"Using existing data from current version: {self.version_id}")
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
                        
                logger.info(f"Copied {len(list(output_dir.iterdir()))} files from test data")
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
                            local_versions.append({
                                "version_id": version_dir.name,
                                "path": raw_data_path,
                                "file_count": file_count
                            })
            
            if not local_versions:
                logger.warning("No local versions found with data.")
                print("\nOptions:")
                print("1. Download new data from cloud (run without --local-only)")
                print("2. Place test data in test_data/raw_data/")
                print("3. Exit")
                
                choice = input("\nSelect option (1-3): ").strip()
                if choice == "1":
                    logger.info("Please run without --local-only flag to download from cloud")
                elif choice == "2":
                    logger.info("Place your test data files in test_data/raw_data/ and run again")
                sys.exit(0)
            else:
                # Show available versions
                print(f"\nFound {len(local_versions)} local version(s) with data:")
                for i, ver in enumerate(local_versions, 1):
                    print(f"{i}. {ver['version_id']} ({ver['file_count']} files)")
                
                print(f"\n{len(local_versions) + 1}. Check cloud for available versions")
                print(f"{len(local_versions) + 2}. Exit")
                
                while True:
                    try:
                        choice = int(input(f"\nSelect version (1-{len(local_versions) + 2}): "))
                        if 1 <= choice <= len(local_versions):
                            # Use selected local version
                            selected = local_versions[choice - 1]
                            logger.info(f"Using data from version: {selected['version_id']}")
                            
                            if not self.dry_run:
                                output_dir.mkdir(parents=True, exist_ok=True)
                                import shutil
                                for item in selected['path'].iterdir():
                                    if item.is_file():
                                        shutil.copy2(item, output_dir)
                                logger.info(f"Copied {len(list(output_dir.iterdir()))} files from {selected['version_id']}")
                            return output_dir
                        elif choice == len(local_versions) + 1:
                            # List cloud versions
                            logger.info("Checking cloud for available versions...")
                            bucket_name = self.config.get("BUCKET_NAME")
                            if bucket_name:
                                cmd = ["gcloud", "storage", "ls", f"gs://{bucket_name}/artifacts/"]
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                if result.returncode == 0:
                                    print("\nCloud versions:")
                                    print(result.stdout)
                                else:
                                    print("Failed to list cloud versions")
                            print("\nTo download from cloud, run without --local-only flag")
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
                "timestamp": datetime.now().isoformat()
            }
            
            with open(metadata_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
                
            # Save detailed download log
            with open(metadata_dir / "download_log.json", 'w') as f:
                json.dump(download_stats, f, indent=2)
                
            logger.info(f"Downloaded {download_stats['files_downloaded']} files")
            logger.info(f"Total size: {download_stats['total_size'] / 1024 / 1024:.2f} MB")
            
            # Check for errors
            if download_stats["errors"]:
                error_count = len(download_stats['errors'])
                logger.error(f"Download failed with {error_count} error(s)")
                for error in download_stats["errors"]:
                    logger.error(f"  - {error}")
                raise RuntimeError(f"Download failed with {error_count} error(s). Check logs for details.")
            
            # Check if any files were actually downloaded
            if download_stats["files_downloaded"] == 0:
                logger.error("No files were downloaded from cloud storage")
                raise RuntimeError("Download failed: No files retrieved from cloud storage")
                
        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id, 
                "download_data",
                {
                    "output_dir": str(output_dir),
                    "files_count": len(list(output_dir.glob("*"))),
                    "completed_at": datetime.now().isoformat()
                }
            )
            
        return output_dir


def run(version_id: str, config: Dict[str, Any], 
        dry_run: bool = False, local_only: bool = False) -> Path:
    """Entry point for the pipeline orchestrator"""
    stage = DownloadDataStage(version_id, config, dry_run, local_only)
    return stage.run()


if __name__ == "__main__":
    # For testing the stage independently
    import click
    from scripts.utils.config_manager import get_config
    
    @click.command()
    @click.option('--version-id', help='Version ID to use')
    @click.option('--dry-run', is_flag=True, help='Preview without downloading')
    @click.option('--local-only', is_flag=True, help='Skip download, use existing data')
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