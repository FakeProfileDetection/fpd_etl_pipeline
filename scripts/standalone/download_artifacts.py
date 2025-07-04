#!/usr/bin/env python3
"""
Download Pipeline Artifacts from GCS
Downloads processed artifacts for local analysis
"""

import sys
import click
import logging
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)
from scripts.utils.cloud_artifact_manager import CloudArtifactManager
from scripts.utils.config_manager import get_config

logger = logging.getLogger(__name__)


@click.command()
@click.option("--version-id", required=True, help="Version ID to download")
@click.option(
    "--stages", multiple=True, help="Specific stages to download (default: all)"
)
@click.option(
    "--output-dir", default="artifacts", help="Output directory (default: artifacts)"
)
@click.option(
    "--include-pii",
    is_flag=True,
    help="Include PII data in download (default: excluded)",
)
@click.option(
    "--force", is_flag=True, help="Force re-download even if files exist locally"
)
def main(
    version_id: str, stages: List[str], output_dir: str, include_pii: bool, force: bool
):
    """Download pipeline artifacts from GCS"""

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

    # Initialize cloud manager
    cloud_manager = CloudArtifactManager(
        project_id=config.get("PROJECT_ID"), bucket_name=config.get("BUCKET_NAME")
    )

    # Check if version exists in cloud
    try:
        manifest = cloud_manager.get_artifact_manifest(version_id)
        if not manifest:
            logger.error(f"No artifacts found for version {version_id} in cloud")
            sys.exit(1)

        click.echo(f"\n📦 Found artifacts for version: {version_id}")
        click.echo(f"   Uploaded at: {manifest.get('upload_timestamp')}")
        click.echo(f"   Total files: {manifest.get('total_files')}")
        click.echo(f"   Stages: {', '.join(manifest.get('stages', []))}")

        if manifest.get("include_pii"):
            click.echo("   ⚠️  Contains PII data")
        else:
            click.echo("   🔒 PII excluded")

    except Exception as e:
        logger.error(f"Failed to check artifacts: {e}")
        sys.exit(1)

    # Check local directory
    local_dir = Path(output_dir) / version_id
    if local_dir.exists() and not force:
        click.echo(f"\n⚠️  Local artifacts already exist at: {local_dir}")
        if not click.confirm("Overwrite existing files?"):
            click.echo("Download cancelled")
            return

    # Filter stages if specified
    artifact_types = stages if stages else None

    # Download artifacts
    click.echo("\n📥 Downloading artifacts...")

    try:
        downloaded = cloud_manager.download_version_artifacts(
            version_id=version_id,
            local_dir=Path(output_dir),
            artifact_types=artifact_types,
            include_pii=include_pii,
            force=force,
        )

        if downloaded:
            # Count downloaded files
            total_files = sum(len(files) for files in downloaded.values())

            click.echo(f"\n✅ Successfully downloaded {total_files} files")
            click.echo(f"\n📁 Artifacts saved to: {local_dir}")

            # Show summary by stage
            click.echo("\nDownloaded:")
            for artifact_type, files in downloaded.items():
                if files:
                    click.echo(f"  - {artifact_type}: {len(files)} files")

            # Update local version info
            vm = VersionManager()
            vm.update_version_metadata(
                version_id,
                {
                    "last_downloaded": datetime.now().isoformat(),
                    "downloaded_from": "gcs",
                },
            )

            click.echo("\n💡 Next steps:")
            click.echo("   - Review the data in the artifacts directory")
            click.echo("   - Run analysis scripts on the downloaded data")
            click.echo("   - Use --version-id with the pipeline to continue processing")

        else:
            click.echo("\n❌ No files were downloaded")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime

    main()
