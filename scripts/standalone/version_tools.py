#!/usr/bin/env python3
"""
Version Management CLI Tool
Provides commands to list, search, delete, and archive pipeline versions
"""

import click
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from tabulate import tabulate
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import EnhancedVersionManager
from scripts.utils.logger_config import setup_pipeline_logging
from scripts.utils.config_manager import get_config

import logging
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Pipeline Version Management Tool"""
    pass


@cli.command()
@click.option('--status', type=click.Choice(['all', 'successful', 'failed', 'archived']), 
              default='all', help='Filter by status')
@click.option('--limit', default=20, help='Maximum number of versions to show')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'simple']), 
              default='table', help='Output format')
def list(status, limit, output_format):
    """List pipeline versions"""
    vm = EnhancedVersionManager()
    versions = vm.list_versions(status=status, limit=limit)
    
    if not versions:
        click.echo(f"No {status} versions found.")
        return
    
    if output_format == 'json':
        click.echo(json.dumps(versions, indent=2))
    elif output_format == 'simple':
        for v in versions:
            status_icon = "✓" if v.get("status") == "completed" else "✗"
            click.echo(f"{status_icon} {v['id']} - {v.get('created_at', 'Unknown date')}")
    else:  # table format
        headers = ["Status", "Version ID", "Created", "Duration", "Stages"]
        rows = []
        
        for v in versions:
            status_icon = "✓" if v.get("status") == "completed" else "✗"
            created = datetime.fromisoformat(v.get("created_at", ""))
            created_str = created.strftime("%Y-%m-%d %H:%M")
            
            # Calculate duration if completed
            duration = "-"
            if v.get("completed_at"):
                completed = datetime.fromisoformat(v["completed_at"])
                duration_sec = (completed - created).total_seconds()
                duration = f"{duration_sec:.1f}s"
            
            # Count completed stages
            stages = v.get("stages", {})
            completed_stages = sum(1 for s in stages.values() if s.get("completed"))
            stages_str = f"{completed_stages}/{len(stages)}" if stages else "0/0"
            
            rows.append([status_icon, v["id"], created_str, duration, stages_str])
        
        click.echo(f"\n{status.upper()} VERSIONS (showing {len(versions)}/{limit})")
        click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@cli.command()
@click.argument('version_id')
@click.option('--export', help='Export to JSON file')
def show(version_id, export):
    """Show detailed information about a version"""
    vm = EnhancedVersionManager()
    version = vm.get_version(version_id)
    
    if not version:
        click.echo(f"Version not found: {version_id}")
        sys.exit(1)
    
    if export:
        vm.export_version_info(version_id, Path(export))
        click.echo(f"Exported version info to: {export}")
        return
    
    # Display version info
    click.echo(f"\nVERSION: {version_id}")
    click.echo("="*60)
    
    # Basic info
    click.echo(f"Status: {version.get('status', 'unknown')}")
    click.echo(f"Created: {version.get('created_at', 'unknown')}")
    if version.get('completed_at'):
        click.echo(f"Completed: {version['completed_at']}")
    
    # Metadata
    if version.get('metadata'):
        click.echo(f"\nMetadata:")
        for key, value in version['metadata'].items():
            click.echo(f"  {key}: {value}")
    
    # Stages
    if version.get('stages'):
        click.echo(f"\nStages:")
        headers = ["Stage", "Status", "Duration", "Output"]
        rows = []
        
        for stage_name, stage_info in version['stages'].items():
            status = "✓" if stage_info.get("completed") else "✗"
            duration = f"{stage_info.get('duration_seconds', 0):.1f}s"
            output = stage_info.get('output_path', '-')
            if output and len(output) > 40:
                output = "..." + output[-37:]
            rows.append([stage_name, status, duration, output])
        
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
    
    # Summary
    if version.get('summary'):
        click.echo(f"\nSummary:")
        for key, value in version['summary'].items():
            click.echo(f"  {key}: {value}")


@cli.command()
@click.argument('version_id')
@click.option('--artifacts', is_flag=True, help='Also delete artifacts directory')
@click.option('--force', is_flag=True, help='Skip confirmation')
def delete(version_id, artifacts, force):
    """Delete a version"""
    vm = EnhancedVersionManager()
    
    # Check if version exists
    version = vm.get_version(version_id)
    if not version:
        click.echo(f"Version not found: {version_id}")
        sys.exit(1)
    
    # Show what will be deleted
    click.echo(f"\nVersion to delete: {version_id}")
    click.echo(f"Status: {version.get('status', 'unknown')}")
    click.echo(f"Created: {version.get('created_at', 'unknown')}")
    
    if artifacts:
        artifacts_dir = Path("artifacts") / version_id
        if artifacts_dir.exists():
            size_mb = sum(f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()) / 1024 / 1024
            click.echo(f"Artifacts: {size_mb:.1f} MB will be deleted")
    
    # Confirm
    if not force:
        if not click.confirm("\nProceed with deletion?"):
            click.echo("Deletion cancelled.")
            return
    
    # Delete
    if vm.delete_version(version_id, delete_artifacts=artifacts):
        click.echo(f"✓ Deleted version: {version_id}")
        if artifacts:
            click.echo("✓ Deleted artifacts")
    else:
        click.echo(f"✗ Failed to delete version: {version_id}")


@cli.command()
@click.argument('version_id')
def archive(version_id):
    """Archive a version (move to archived storage)"""
    vm = EnhancedVersionManager()
    
    if vm.archive_version(version_id):
        click.echo(f"✓ Archived version: {version_id}")
    else:
        click.echo(f"✗ Failed to archive version: {version_id}")


@cli.command()
@click.option('--days', default=7, help='Delete failed versions older than N days')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without deleting')
def cleanup(days, dry_run):
    """Clean up old failed versions"""
    vm = EnhancedVersionManager()
    
    # Find versions to delete
    cutoff_date = datetime.now() - timedelta(days=days)
    failed_versions = vm.list_versions(status="failed", limit=1000)
    
    to_delete = []
    for version in failed_versions:
        created_at = datetime.fromisoformat(version["created_at"])
        if created_at < cutoff_date:
            to_delete.append(version)
    
    if not to_delete:
        click.echo(f"No failed versions older than {days} days found.")
        return
    
    # Show what will be deleted
    click.echo(f"\nFound {len(to_delete)} failed versions older than {days} days:")
    for v in to_delete[:10]:  # Show first 10
        click.echo(f"  - {v['id']} (created {v['created_at']})")
    if len(to_delete) > 10:
        click.echo(f"  ... and {len(to_delete) - 10} more")
    
    if dry_run:
        click.echo("\nDRY RUN - no versions deleted")
        return
    
    # Confirm
    if not click.confirm(f"\nDelete {len(to_delete)} failed versions?"):
        click.echo("Cleanup cancelled.")
        return
    
    # Delete
    deleted_count = vm.cleanup_failed_versions(days=days)
    click.echo(f"\n✓ Deleted {deleted_count} failed versions")


@cli.command()
@click.option('--stage-failed', help='Find versions where a specific stage failed')
@click.option('--days-ago', type=int, help='Find versions created in the last N days')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
def search(stage_failed, days_ago, output_format):
    """Search for versions matching criteria"""
    vm = EnhancedVersionManager()
    
    criteria = {}
    if stage_failed:
        criteria['stage_failed'] = stage_failed
    if days_ago:
        criteria['created_after'] = datetime.now() - timedelta(days=days_ago)
    
    if not criteria:
        click.echo("Please specify at least one search criterion")
        return
    
    results = vm.search_versions(**criteria)
    
    if not results:
        click.echo("No versions found matching criteria")
        return
    
    click.echo(f"\nFound {len(results)} matching versions:")
    
    if output_format == 'json':
        click.echo(json.dumps(results, indent=2))
    else:
        # Table format
        headers = ["Version ID", "Status", "Created", "Failed Stage"]
        rows = []
        
        for v in results:
            created = datetime.fromisoformat(v.get("created_at", ""))
            created_str = created.strftime("%Y-%m-%d %H:%M")
            
            # Find failed stages
            failed_stages = []
            for stage_name, stage_info in v.get("stages", {}).items():
                if not stage_info.get("completed", True):
                    failed_stages.append(stage_name)
            
            rows.append([
                v["id"], 
                v.get("status", "unknown"),
                created_str,
                ", ".join(failed_stages) or "-"
            ])
        
        click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@cli.command()
def stats():
    """Show version statistics"""
    vm = EnhancedVersionManager()
    
    # Get counts
    successful = len(vm.list_versions(status="successful", limit=1000))
    failed = len(vm.list_versions(status="failed", limit=1000))
    
    # Get current version
    current_id = vm.get_current_version_id()
    
    click.echo("\nVERSION STATISTICS")
    click.echo("="*40)
    click.echo(f"Successful versions: {successful}")
    click.echo(f"Failed versions: {failed}")
    click.echo(f"Total versions: {successful + failed}")
    
    if current_id:
        click.echo(f"\nCurrent version: {current_id}")
        current = vm.get_version(current_id)
        if current:
            click.echo(f"Created: {current.get('created_at', 'unknown')}")
    
    # Check disk usage
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        total_size = 0
        version_count = 0
        for version_dir in artifacts_dir.iterdir():
            if version_dir.is_dir():
                version_count += 1
                for f in version_dir.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
        
        click.echo(f"\nArtifacts storage:")
        click.echo(f"  Versions with artifacts: {version_count}")
        click.echo(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    
    # Show recent failures
    recent_failed = vm.list_versions(status="failed", limit=5)
    if recent_failed:
        click.echo(f"\nRecent failures:")
        for v in recent_failed:
            stages = v.get("stages", {})
            failed_stages = [s for s, info in stages.items() if not info.get("completed", True)]
            click.echo(f"  - {v['id'][:30]}... ({', '.join(failed_stages) or 'unknown'})")


def delete_version_completely(version_id: str, config: Dict[str, Any], vm: EnhancedVersionManager) -> bool:
    """
    Completely delete a version including all artifacts (local and cloud)
    Used by both purge and cleanup tools
    """
    # Delete from version tracking
    vm.delete_version(version_id, delete_artifacts=False)
    
    # Delete local artifacts
    artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts")) / version_id
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    
    # Delete cloud artifacts if configured
    bucket_name = config.get("GCS_BUCKET_NAME")
    if bucket_name:
        try:
            cmd = ["gsutil", "-m", "rm", "-r", f"gs://{bucket_name}/artifacts/{version_id}/"]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            # Cloud deletion failed, but continue
            pass
    
    return True


if __name__ == "__main__":
    # Set up logging (minimal for CLI tool)
    logging.basicConfig(level=logging.WARNING)
    
    cli()