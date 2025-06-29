#!/usr/bin/env python3
"""
Upload Pipeline Artifacts to GCS
Uploads processed artifacts for team collaboration
"""

import sys
import click
import logging
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager
from scripts.utils.cloud_artifact_manager import CloudArtifactManager
from scripts.utils.config_manager import get_config

logger = logging.getLogger(__name__)


def list_artifacts(version_id: str, artifacts_dir: Path) -> dict:
    """List all artifacts for a version"""
    version_dir = artifacts_dir / version_id
    
    if not version_dir.exists():
        return {}
    
    artifacts = {
        "raw_data": list(version_dir.glob("raw_data/**/*")),
        "cleaned_data": list(version_dir.glob("cleaned_data/**/*")),
        "keypairs": list(version_dir.glob("keypairs/**/*")),
        "features": list(version_dir.glob("features/**/*")),
        "reports": list(version_dir.glob("reports/**/*")),
        "etl_metadata": list(version_dir.glob("etl_metadata/**/*"))
    }
    
    # Count files and calculate sizes
    for key, files in artifacts.items():
        file_list = [f for f in files if f.is_file()]
        artifacts[key] = {
            "files": file_list,
            "count": len(file_list),
            "size_mb": sum(f.stat().st_size for f in file_list) / (1024 * 1024)
        }
    
    return artifacts


@click.command()
@click.option('--version-id', required=True, help='Version ID to upload')
@click.option('--stages', multiple=True, 
              help='Specific stages to upload (default: all)')
@click.option('--include-pii', is_flag=True, 
              help='Include PII data in upload (default: excluded)')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be uploaded without uploading')
@click.option('--force', is_flag=True,
              help='Force re-upload even if artifacts already exist')
def main(version_id: str, stages: List[str], include_pii: bool, 
         dry_run: bool, force: bool):
    """Upload pipeline artifacts to GCS for team sharing"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = get_config()
    
    # Validate cloud config
    if not config.validate_cloud_config():
        logger.error("Cloud configuration is incomplete. Check PROJECT_ID and BUCKET_NAME.")
        sys.exit(1)
    
    # Get version info
    vm = VersionManager()
    version_info = vm.get_version(version_id)
    
    if not version_info:
        logger.error(f"Version {version_id} not found")
        sys.exit(1)
    
    # Initialize cloud manager
    cloud_manager = CloudArtifactManager(
        project_id=config.get("PROJECT_ID"),
        bucket_name=config.get("BUCKET_NAME")
    )
    
    # Check if artifacts already exist
    if not force and not dry_run:
        try:
            manifest = cloud_manager.get_artifact_manifest(version_id)
            if manifest:
                click.echo(f"\n⚠️  Artifacts already exist for version {version_id}")
                click.echo(f"   Uploaded at: {manifest.get('upload_timestamp')}")
                click.echo(f"   Total files: {manifest.get('total_files')}")
                click.echo("\nUse --force to re-upload")
                return
        except:
            # No manifest found, proceed with upload
            pass
    
    # List local artifacts
    artifacts_dir = Path("artifacts")
    artifacts = list_artifacts(version_id, artifacts_dir)
    
    # Display summary
    click.echo(f"\n📦 Artifacts for version: {version_id}")
    click.echo("=" * 60)
    
    total_files = 0
    total_size = 0
    
    # Filter stages if specified
    if stages:
        artifacts = {k: v for k, v in artifacts.items() if k in stages}
    
    for stage, info in artifacts.items():
        if info["count"] > 0:
            click.echo(f"\n{stage}:")
            click.echo(f"  Files: {info['count']}")
            click.echo(f"  Size: {info['size_mb']:.2f} MB")
            total_files += info["count"]
            total_size += info["size_mb"]
    
    click.echo(f"\nTotal: {total_files} files, {total_size:.2f} MB")
    
    if include_pii:
        click.echo("\n⚠️  WARNING: PII data WILL be included in upload")
    else:
        click.echo("\n🔒 PII data will be excluded (default)")
    
    if dry_run:
        click.echo("\n🔍 DRY RUN - No files will be uploaded")
        return
    
    # Confirm upload
    if not click.confirm("\nProceed with upload?"):
        click.echo("Upload cancelled")
        return
    
    # Upload artifacts
    click.echo("\n📤 Uploading artifacts...")
    
    try:
        uploaded_count = 0
        
        with click.progressbar(length=total_files, label='Uploading files') as bar:
            for stage, info in artifacts.items():
                for file_path in info["files"]:
                    # Calculate relative path within version directory
                    relative_path = file_path.relative_to(artifacts_dir / version_id)
                    
                    # Check PII patterns
                    if not include_pii and cloud_manager._is_pii_file(str(relative_path)):
                        continue
                    
                    # Upload file
                    success = cloud_manager.upload_artifact(
                        local_path=file_path,
                        artifact_path=f"artifacts/{version_id}/{relative_path}",
                        version_id=version_id,
                        metadata={
                            "stage": stage,
                            "original_path": str(relative_path)
                        }
                    )
                    
                    if success:
                        uploaded_count += 1
                    
                    bar.update(1)
        
        # Create manifest
        manifest_created = cloud_manager.create_artifact_manifest(
            version_id=version_id,
            artifacts_info={
                "stages": list(artifacts.keys()),
                "total_files": uploaded_count,
                "total_size_mb": total_size,
                "include_pii": include_pii
            }
        )
        
        if manifest_created:
            click.echo(f"\n✅ Successfully uploaded {uploaded_count} files")
            click.echo(f"\n📋 Manifest created at: gs://{config.get('BUCKET_NAME')}/artifacts/{version_id}/artifact_manifest.json")
        else:
            click.echo("\n⚠️  Files uploaded but manifest creation failed")
            
        # Update version info
        vm.update_version_metadata(version_id, {
            "artifacts_uploaded": True,
            "upload_timestamp": datetime.now().isoformat(),
            "uploaded_files": uploaded_count
        })
        
        click.echo("\n💡 Team members can download these artifacts with:")
        click.echo(f"   python scripts/standalone/download_artifacts.py --version-id {version_id}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime
    main()