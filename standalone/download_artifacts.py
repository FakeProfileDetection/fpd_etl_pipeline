#!/usr/bin/env python3
"""
Download artifacts from cloud storage
Used by team members to get processed data
"""

import click
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager
from scripts.utils.config_manager import get_config
from scripts.utils.cloud_artifact_manager import CloudArtifactManager, create_pii_filter

logger = logging.getLogger(__name__)


@click.command()
@click.option('--version', default='latest',
              help='Version to download (default: latest)')
@click.option('--stages', '-s', multiple=True,
              help='Specific stages to download (default: all)')
@click.option('--artifact-types', '-t', multiple=True,
              help='Specific artifact types to download')
@click.option('--include-pii', is_flag=True, default=False,
              help='Include PII data in download (default: False)')
@click.option('--output-dir', type=Path,
              help='Output directory (default: artifacts/{version_id})')
@click.option('--force', is_flag=True,
              help='Force re-download even if cached locally')
@click.option('--list-only', is_flag=True,
              help='List available artifacts without downloading')
def main(version: str, stages: List[str], artifact_types: List[str],
         include_pii: bool, output_dir: Optional[Path], 
         force: bool, list_only: bool):
    """
    Download artifacts from cloud storage.
    
    By default, PII data (demographics, consent) is EXCLUDED.
    Use --include-pii only if you need this data and have proper authorization.
    
    Examples:
        # Download latest artifacts (no PII)
        python download_artifacts.py
        
        # Download specific version
        python download_artifacts.py --version 2024-01-15_14-30-00_macbook-alice
        
        # Download specific stages
        python download_artifacts.py -s cleaned_data -s features
        
        # List available artifacts
        python download_artifacts.py --list-only
        
        # Download with PII (requires confirmation)
        python download_artifacts.py --include-pii
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
    
    # Get version
    vm = VersionManager()
    
    if version == 'latest' or version == 'current':
        version_info = vm.get_latest_complete_version()
        if not version_info:
            click.echo("❌ No complete versions found")
            click.echo("    Available versions:")
            for v in vm.list_versions(limit=5):
                click.echo(f"    - {v['version_id']} ({v.get('status', 'unknown')})")
            sys.exit(1)
        version_id = version_info['version_id']
    else:
        version_id = version
        version_info = vm.get_version(version_id)
        if not version_info:
            click.echo(f"❌ Version not found: {version_id}")
            click.echo("    Use --list-only to see available versions")
            sys.exit(1)
    
    click.echo(f"📥 {'Listing' if list_only else 'Downloading'} artifacts for version: {version_id}")
    
    # Show version info
    if version_info:
        created = version_info.get('created_at', 'unknown')
        status = version_info.get('status', 'unknown')
        stages_run = version_info.get('stages', {}).keys()
        click.echo(f"   Created: {created}")
        click.echo(f"   Status: {status}")
        click.echo(f"   Stages: {', '.join(stages_run)}")
    
    # PII warning
    if include_pii:
        click.echo("\n⚠️  WARNING: You've requested to include PII data")
        if not list_only and not click.confirm("Are you authorized to access PII data?"):
            return
    
    # Initialize cloud manager
    cloud_mgr = CloudArtifactManager(
        version_id=version_id,
        bucket_name=config.get("BUCKET_NAME"),
        local_cache_dir=output_dir if output_dir else Path(".artifact_cache")
    )
    
    # Check if cloud is available
    if not cloud_mgr.cloud_enabled:
        click.echo("❌ Cloud storage not available. Check your Google Cloud credentials.")
        sys.exit(1)
    
    # Create PII filter
    pii_filter = None
    if not include_pii:
        pii_patterns = config.get_pii_patterns()
        pii_filter = create_pii_filter(pii_patterns)
    
    # List artifacts
    if list_only:
        artifacts = cloud_mgr.list_artifacts()
        
        # Filter by criteria
        if stages:
            artifacts = [a for a in artifacts if a['stage'] in stages]
        if artifact_types:
            artifacts = [a for a in artifacts if a['artifact_type'] in artifact_types]
        if pii_filter:
            artifacts = [a for a in artifacts if pii_filter(a['local_name'])]
        
        if not artifacts:
            click.echo("No artifacts found matching criteria")
            return
        
        # Group by stage
        by_stage = {}
        total_size = 0
        for artifact in artifacts:
            stage = artifact['stage']
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(artifact)
            total_size += artifact['size_bytes']
        
        # Display
        click.echo(f"\n📦 Found {len(artifacts)} artifacts ({total_size / 1024 / 1024:.2f} MB)")
        for stage, stage_artifacts in sorted(by_stage.items()):
            click.echo(f"\n{stage}:")
            for a in stage_artifacts[:5]:  # Show first 5
                size_mb = a['size_bytes'] / 1024 / 1024
                click.echo(f"  - {a['local_name']} ({size_mb:.2f} MB) - {a['artifact_type']}")
            if len(stage_artifacts) > 5:
                click.echo(f"  ... and {len(stage_artifacts) - 5} more")
        
        return
    
    # Download artifacts
    click.echo("\nDownloading artifacts...")
    
    # Determine what to download
    stages_to_download = stages if stages else None
    
    downloaded_files = {}
    failed_downloads = []
    
    # Download by stage
    all_stages = cloud_mgr.list_artifacts()
    stages_available = list(set(a['stage'] for a in all_stages))
    
    for stage in stages_available:
        if stages_to_download and stage not in stages_to_download:
            continue
        
        click.echo(f"\n📂 Stage: {stage}")
        
        try:
            stage_files = cloud_mgr.download_stage_artifacts(
                stage=stage,
                artifact_types=artifact_types if artifact_types else None,
                pii_filter=pii_filter
            )
            
            downloaded_files.update(stage_files)
            
            for artifact_id, local_path in stage_files.items():
                click.echo(f"  ✓ {local_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to download stage {stage}: {e}")
            failed_downloads.append(stage)
    
    # Summary
    click.echo(f"\n✅ Download complete:")
    click.echo(f"   Downloaded: {len(downloaded_files)} files")
    if failed_downloads:
        click.echo(f"   Failed stages: {', '.join(failed_downloads)}")
    
    # Show output location
    if downloaded_files:
        first_file = next(iter(downloaded_files.values()))
        output_base = first_file.parent.parent
        click.echo(f"\n📁 Files saved to: {output_base}")
        
        # Show how to use the data
        click.echo("\n💡 To use this data:")
        click.echo(f"   cd {output_base}")
        click.echo("   python")
        click.echo("   >>> import pandas as pd")
        click.echo("   >>> df = pd.read_parquet('cleaned_data/data.parquet')")
    
    # Update local versions.json to track downloaded version
    if downloaded_files and not output_dir:
        vm.versions_data['last_downloaded'] = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'stages': list(set(a['stage'] for a in cloud_mgr.list_artifacts() if a['id'] in downloaded_files))
        }
        vm._save_versions()


if __name__ == "__main__":
    from datetime import datetime
    main()

