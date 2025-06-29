#!/usr/bin/env python3
"""
Upload existing local artifacts to cloud storage
Useful for uploading after local development and review
"""

import click
import logging
import sys
from pathlib import Path
from typing import List, Optional
from fnmatch import fnmatch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager
from scripts.utils.config_manager import get_config
from scripts.utils.cloud_artifact_manager import CloudArtifactManager, create_pii_filter

logger = logging.getLogger(__name__)


def get_latest_local_version() -> Optional[str]:
    """Find the most recently modified local version"""
    artifacts_dir = Path('artifacts')
    if not artifacts_dir.exists():
        return None
    
    versions = [d for d in artifacts_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not versions:
        return None
    
    # Sort by modification time
    latest = max(versions, key=lambda d: d.stat().st_mtime)
    return latest.name


def find_local_artifacts(version_id: str, stages: List[str],
                        artifact_types: List[str], 
                        include_pii: bool,
                        pii_patterns: List[str]) -> List[Path]:
    """Find all artifacts matching criteria"""
    artifacts_dir = Path('artifacts') / version_id
    if not artifacts_dir.exists():
        return []
    
    artifacts = []
    pii_filter = create_pii_filter(pii_patterns) if not include_pii else None
    
    for path in artifacts_dir.rglob('*'):
        if not path.is_file():
            continue
            
        # Skip hidden files and manifests
        if path.name.startswith('.') or path.name == 'artifact_manifest.json':
            continue
        
        # Skip system files
        if path.suffix in ['.pyc', '.log', '.tmp']:
            continue
        
        # Check PII filter
        if pii_filter and not pii_filter(path.name):
            logger.debug(f"Excluding PII file: {path.name}")
            continue
        
        # Check stage filter
        if stages:
            path_str = str(path.relative_to(artifacts_dir))
            if not any(stage in path_str for stage in stages):
                continue
        
        # Check artifact type filter
        if artifact_types:
            if not any(atype in str(path) for atype in artifact_types):
                continue
        
        artifacts.append(path)
    
    return sorted(artifacts)


def parse_artifact_path(artifact_path: Path, version_id: str) -> tuple[str, str]:
    """Parse artifact path to determine stage and type"""
    # Get relative path from version directory
    try:
        rel_path = artifact_path.relative_to(Path('artifacts') / version_id)
        parts = rel_path.parts
        
        if len(parts) >= 2:
            # Typical structure: stage/type/filename
            stage = parts[0]
            artifact_type = parts[1] if len(parts) > 2 else 'general'
        else:
            # Fallback
            stage = 'general'
            artifact_type = 'file'
            
        return stage, artifact_type
    except ValueError:
        # Not in expected structure
        return 'unknown', 'file'


@click.command()
@click.option('--version-id', help='Version to upload (default: latest local)')
@click.option('--stages', '-s', multiple=True,
              help='Specific stages to upload (default: all)')
@click.option('--artifact-types', '-t', multiple=True,
              help='Specific artifact types to upload')
@click.option('--include-pii', is_flag=True, default=False,
              help='Include PII files in upload (default: False)')
@click.option('--dry-run', is_flag=True,
              help='Show what would be uploaded without uploading')
@click.option('--force', is_flag=True,
              help='Re-upload even if already exists in cloud')
def main(version_id: Optional[str], stages: List[str],
         artifact_types: List[str], include_pii: bool,
         dry_run: bool, force: bool):
    """
    Upload existing local artifacts to cloud storage.
    
    This is useful when you've run the pipeline locally and decided
    the results are good to share with the team.
    
    Examples:
        # Upload latest version (excluding PII by default)
        python upload_artifacts.py
        
        # Upload specific stages only
        python upload_artifacts.py -s cleaned_data -s features
        
        # See what would be uploaded
        python upload_artifacts.py --dry-run
        
        # Upload everything including PII (requires confirmation)
        python upload_artifacts.py --include-pii
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = get_config()
    
    # Validate cloud configuration
    if not config.get("BUCKET_NAME"):
        click.echo("❌ No bucket configured. Please set BUCKET_NAME in config/.env.local")
        sys.exit(1)
    
    # Get version if not specified
    if not version_id:
        version_id = get_latest_local_version()
        if not version_id:
            click.echo("❌ No local artifacts found to upload")
            click.echo("    Run the pipeline first: python scripts/pipeline/run_pipeline.py")
            sys.exit(1)
    
    click.echo(f"📤 Preparing to upload artifacts for version: {version_id}")
    
    # Check if version exists
    artifacts_dir = Path('artifacts') / version_id
    if not artifacts_dir.exists():
        click.echo(f"❌ Version directory not found: {artifacts_dir}")
        sys.exit(1)
    
    # Find local artifacts
    pii_patterns = config.get_pii_patterns()
    local_artifacts = find_local_artifacts(
        version_id, stages, artifact_types, include_pii, pii_patterns
    )
    
    if not local_artifacts:
        click.echo("❌ No artifacts found matching criteria")
        sys.exit(1)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in local_artifacts) / 1024 / 1024
    
    # Show what will be uploaded
    click.echo(f"\nFound {len(local_artifacts)} artifacts ({total_size:.2f} MB)")
    
    # Check for PII files
    if include_pii:
        pii_filter = create_pii_filter(pii_patterns)
        pii_files = [f for f in local_artifacts if not pii_filter(f.name)]
        if pii_files:
            click.echo(f"⚠️  WARNING: {len(pii_files)} files may contain PII:")
            for f in pii_files[:5]:  # Show first 5
                click.echo(f"    - {f.name}")
            if len(pii_files) > 5:
                click.echo(f"    ... and {len(pii_files) - 5} more")
            
            if not dry_run and not click.confirm("\nContinue with PII upload?"):
                return
    
    # Show sample of files
    click.echo("\nFiles to upload:")
    for i, artifact in enumerate(local_artifacts[:10]):
        rel_path = artifact.relative_to(Path('artifacts'))
        size_mb = artifact.stat().st_size / 1024 / 1024
        click.echo(f"  - {rel_path} ({size_mb:.2f} MB)")
    if len(local_artifacts) > 10:
        click.echo(f"  ... and {len(local_artifacts) - 10} more files")
    
    if dry_run:
        click.echo("\n🔍 DRY RUN - No files will be uploaded")
        return
    
    # Confirm upload
    if not click.confirm(f"\nUpload {len(local_artifacts)} artifacts to cloud?"):
        click.echo("Upload cancelled")
        return
    
    # Initialize cloud manager
    cloud_mgr = CloudArtifactManager(
        version_id=version_id,
        bucket_name=config.get("BUCKET_NAME")
    )
    
    # Check if cloud is available
    if not cloud_mgr.cloud_enabled:
        click.echo("❌ Cloud storage not available. Check your Google Cloud credentials.")
        sys.exit(1)
    
    # Upload artifacts
    uploaded = 0
    failed = 0
    skipped = 0
    
    with click.progressbar(local_artifacts, label='Uploading artifacts') as artifacts:
        for artifact_path in artifacts:
            try:
                # Determine metadata from path
                stage, artifact_type = parse_artifact_path(artifact_path, version_id)
                
                # Check if already uploaded
                artifact_name = artifact_path.name
                artifact_id = f"{stage}_{artifact_type}_{artifact_name}"
                
                if not force and artifact_id in cloud_mgr.manifest.get('artifacts', {}):
                    logger.debug(f"Skipping {artifact_name} - already uploaded")
                    skipped += 1
                    continue
                
                # Upload
                cloud_mgr.upload_artifact(
                    local_path=artifact_path,
                    artifact_type=artifact_type,
                    stage=stage,
                    description=f"Uploaded via standalone script",
                    force=force
                )
                uploaded += 1
                
            except Exception as e:
                logger.error(f"Failed to upload {artifact_path.name}: {e}")
                failed += 1
    
    # Summary
    click.echo(f"\n✅ Upload complete:")
    click.echo(f"   - Uploaded: {uploaded}")
    click.echo(f"   - Skipped: {skipped}")
    click.echo(f"   - Failed: {failed}")
    
    if uploaded > 0:
        # Update version manager
        vm = VersionManager()
        vm.update_stage_info(version_id, "artifacts_uploaded", {
            "timestamp": datetime.now().isoformat(),
            "uploaded_count": uploaded,
            "total_size_mb": total_size
        })
        
        click.echo("\n📝 Updated versions.json - remember to commit and push:")
        click.echo("    git add versions.json")
        click.echo(f"    git commit -m 'Upload artifacts for {version_id}'")
        click.echo("    git push")
        
        # Show manifest location
        manifest_url = f"gs://{config.get('BUCKET_NAME')}/artifacts/{version_id}/artifact_manifest.json"
        click.echo(f"\n📋 Artifact manifest: {manifest_url}")


if __name__ == "__main__":
    from datetime import datetime
    main()

