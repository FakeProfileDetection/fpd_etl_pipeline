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

from scripts.utils.enhanced_version_manager import (  # noqa: E402
    EnhancedVersionManager as VersionManager,
)
from scripts.utils.cloud_artifact_manager import CloudArtifactManager  # noqa: E402
from scripts.utils.config_manager import get_config  # noqa: E402

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

    # Initialize cloud manager for listing purposes (we'll need version_id later)
    # For now, just validate config and check cloud availability
    bucket_name = config.get("BUCKET_NAME")

    # Initialize cloud manager with version_id
    cloud_manager = CloudArtifactManager(
        version_id=version_id, bucket_name=bucket_name
    )
    
    # Check if version exists in cloud
    try:
        # Get version info from local tracking
        vm = VersionManager()
        version_info = vm.get_version(version_id)
        if not version_info:
            logger.error(f"Version {version_id} not found")
            sys.exit(1)
        
        click.echo(f"\n📦 Version: {version_id}")
        if version_info.get("summary", {}).get("artifacts_uploaded"):
            click.echo("   ☁️  Artifacts available in cloud")
        else:
            logger.error(f"No artifacts uploaded for version {version_id}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to check version: {e}")
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
        # Download all stage artifacts
        downloaded = {}
        stages_to_download = artifact_types if artifact_types else None
        
        # Get list of available stages from version info
        available_stages = list(version_info.get("stages", {}).keys())
        if stages_to_download:
            # Filter to requested stages
            available_stages = [s for s in available_stages if s in stages_to_download]
        
        for stage in available_stages:
            try:
                stage_files = cloud_manager.download_stage_artifacts(
                    stage_name=stage,
                    local_dir=Path(output_dir) / version_id,
                    force=force
                )
                if stage_files:
                    downloaded[stage] = stage_files
            except Exception as e:
                logger.warning(f"Failed to download {stage}: {e}")

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
            vm.update_version(
                version_id,
                summary={
                    "last_downloaded": datetime.now().isoformat(),
                    "downloaded_from": "gcs",
                }
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
