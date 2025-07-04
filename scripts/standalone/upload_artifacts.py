#!/usr/bin/env python3
"""
Upload Pipeline Artifacts to GCS
Creates a tar.gz archive of artifacts and uploads to cloud storage
"""

import logging
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config_manager import get_config
from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


def create_tarball(
    version_id: str, artifacts_dir: Path, include_pii: bool = False
) -> Optional[Path]:
    """Create tar.gz archive of version artifacts"""
    version_dir = artifacts_dir / version_id

    if not version_dir.exists():
        logger.error(f"Version directory not found: {version_dir}")
        return None

    # Create tar.gz file
    tarball_name = f"{version_id}.tar.gz"
    tarball_path = artifacts_dir / tarball_name

    logger.info(f"Creating archive: {tarball_name}")

    try:
        # Define exclusion patterns for PII
        exclude_patterns = []
        if not include_pii:
            exclude_patterns = [
                "*metadata/metadata.csv",
                "*demographics.json",
                "*consent.json",
                "*email*",
                "*start_time.json",
            ]

        def should_exclude(tarinfo):
            """Filter function for tarfile to exclude PII files"""
            if not exclude_patterns:
                return False

            for pattern in exclude_patterns:
                if pattern.replace("*", "") in tarinfo.name:
                    logger.debug(f"Excluding PII file: {tarinfo.name}")
                    return True
            return False

        # Create archive
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(
                version_dir,
                arcname=version_id,
                filter=lambda x: None if should_exclude(x) else x,
            )

        size_mb = tarball_path.stat().st_size / 1024 / 1024
        logger.info(f"Archive created: {size_mb:.2f} MB")

        return tarball_path

    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        if tarball_path.exists():
            tarball_path.unlink()
        return None


def check_existing_version(bucket_name: str, version_id: str) -> bool:
    """Check if version already exists in cloud"""
    tarball_name = f"{version_id}.tar.gz"
    check_cmd = [
        "gcloud",
        "storage",
        "ls",
        f"gs://{bucket_name}/artifacts/{tarball_name}",
    ]

    result = subprocess.run(check_cmd, capture_output=True, text=True)
    return result.returncode == 0


def upload_tarball(tarball_path: Path, bucket_name: str) -> bool:
    """Upload tarball to cloud storage"""
    tarball_name = tarball_path.name

    upload_cmd = [
        "gcloud",
        "storage",
        "cp",
        str(tarball_path),
        f"gs://{bucket_name}/artifacts/{tarball_name}",
    ]

    logger.info(f"Uploading to gs://{bucket_name}/artifacts/{tarball_name}")
    result = subprocess.run(upload_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Upload failed: {result.stderr}")
        return False

    logger.info("✅ Upload completed successfully")
    return True


@click.command()
@click.option("--version-id", required=True, help="Version ID to upload")
@click.option(
    "--include-pii", is_flag=True, help="Include PII data in upload (default: excluded)"
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be uploaded without uploading"
)
@click.option(
    "--force", is_flag=True, help="Force re-upload even if version already exists"
)
def main(version_id: str, include_pii: bool, dry_run: bool, force: bool):
    """
    Upload pipeline artifacts to cloud storage as tar.gz archive.

    By default, PII data is excluded from the upload.
    Only new versions can be uploaded - existing versions cannot be modified.
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = get_config()

    # Validate cloud config
    if not config.validate_cloud_config():
        logger.error(
            "Cloud configuration is incomplete. Check PROJECT_ID and BUCKET_NAME."
        )
        sys.exit(1)

    bucket_name = config.get("BUCKET_NAME")

    # Get version info
    vm = VersionManager()
    version_info = vm.get_version(version_id)

    if not version_info:
        logger.error(f"Version {version_id} not found")
        sys.exit(1)

    # Check if version already exists in cloud
    if not force and check_existing_version(bucket_name, version_id):
        logger.error(f"Version {version_id} already exists in cloud")
        logger.info("Per policy: Existing versions cannot be modified")
        logger.info("If you need to upload different artifacts, create a new version")
        sys.exit(1)

    # Display upload info
    artifacts_dir = Path("artifacts")
    version_dir = artifacts_dir / version_id

    if not version_dir.exists():
        logger.error(f"No artifacts found for version {version_id}")
        sys.exit(1)

    click.echo(f"\n📦 Preparing to upload version: {version_id}")
    click.echo("=" * 60)

    # Count files and size
    total_files = sum(1 for _ in version_dir.rglob("*") if _.is_file())
    total_size_mb = (
        sum(f.stat().st_size for f in version_dir.rglob("*") if f.is_file())
        / 1024
        / 1024
    )

    click.echo(f"Total files: {total_files}")
    click.echo(f"Total size: {total_size_mb:.2f} MB")

    if include_pii:
        click.echo("\n⚠️  WARNING: PII data WILL be included in upload")
    else:
        click.echo("\n🔒 PII data will be excluded (default)")

    if dry_run:
        click.echo("\n🔍 DRY RUN - No files will be uploaded")

        # Show what would be excluded
        if not include_pii:
            click.echo("\nFiles that would be excluded:")
            excluded = 0
            for pattern in [
                "demographics.json",
                "consent.json",
                "metadata.csv",
                "email",
                "start_time.json",
            ]:
                for f in version_dir.rglob(f"*{pattern}*"):
                    if f.is_file():
                        click.echo(f"  - {f.relative_to(version_dir)}")
                        excluded += 1
            click.echo(f"\nTotal excluded: {excluded} files")

        return

    # Confirm upload
    if not click.confirm("\nProceed with upload?"):
        click.echo("Upload cancelled")
        return

    # Create tarball
    click.echo("\n📦 Creating archive...")
    tarball_path = create_tarball(version_id, artifacts_dir, include_pii)

    if not tarball_path:
        click.echo("❌ Failed to create archive")
        sys.exit(1)

    try:
        # Upload tarball
        click.echo("\n📤 Uploading to cloud storage...")
        if upload_tarball(tarball_path, bucket_name):
            click.echo("\n✅ Upload completed successfully!")

            # Update version info
            vm.update_version(
                version_id,
                {
                    "artifacts_uploaded": True,
                    "upload_timestamp": datetime.now().isoformat(),
                    "upload_included_pii": include_pii,
                },
            )
        else:
            click.echo("\n❌ Upload failed")
            sys.exit(1)

    finally:
        # Clean up local tarball
        if tarball_path.exists():
            tarball_path.unlink()
            logger.debug("Cleaned up local archive")


if __name__ == "__main__":
    main()
