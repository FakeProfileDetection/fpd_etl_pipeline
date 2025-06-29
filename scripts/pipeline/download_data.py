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

from scripts.utils.version_manager import VersionManager
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
            return []
            
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
            logger.info("Local-only mode: Skipping download, using existing data")
            if not output_dir.exists():
                logger.warning(f"No existing data found at {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
            
        # Validate authentication
        if not self.validate_gcs_auth():
            raise RuntimeError("GCS authentication failed")
            
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
            
            if download_stats["errors"]:
                logger.warning(f"Encountered {len(download_stats['errors'])} errors during download")
                
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