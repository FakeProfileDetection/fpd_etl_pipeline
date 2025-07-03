#!/usr/bin/env python3
"""
Development Version Cleanup Tool
A safer alternative to purging - allows selective cleanup of development versions
"""

import click
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import EnhancedVersionManager as VersionManager
from scripts.utils.config_manager import get_config
from scripts.standalone.version_tools import delete_version_completely


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def get_artifact_size(version_id: str, config: Dict[str, Any]) -> float:
    """Get size of artifacts for a version in MB"""
    artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts"))
    version_dir = artifacts_dir / version_id
    
    if not version_dir.exists():
        return 0.0
    
    total_size = sum(f.stat().st_size for f in version_dir.rglob('*') if f.is_file())
    return total_size / 1024 / 1024  # Convert to MB


def cleanup_by_age(days: int, dry_run: bool = False, include_successful: bool = False):
    """Delete versions older than specified days"""
    vm = VersionManager()
    config = get_config().config
    
    cutoff_date = datetime.now() - timedelta(days=days)
    versions_to_delete = []
    
    # Get all versions
    all_versions = vm.list_versions(status='all', limit=None)
    
    for version in all_versions:
        created_at = datetime.fromisoformat(version['created_at'])
        if created_at < cutoff_date:
            if version['status'] == 'completed' and not include_successful:
                continue  # Skip successful versions unless explicitly included
            
            version_id = version.get('version_id') or version.get('id')
            size = get_artifact_size(version_id, config)
            versions_to_delete.append({
                'version_id': version_id,
                'status': version['status'],
                'created_at': created_at,
                'size_mb': size
            })
    
    if not versions_to_delete:
        print(f"{Colors.GREEN}No versions older than {days} days found.{Colors.RESET}")
        return
    
    # Display what will be deleted
    print(f"\n{Colors.YELLOW}Found {len(versions_to_delete)} versions older than {days} days:{Colors.RESET}")
    total_size = 0
    for v in versions_to_delete:
        status_color = Colors.GREEN if v['status'] == 'completed' else Colors.RED
        print(f"  • {v['version_id']} - {status_color}{v['status']}{Colors.RESET} - "
              f"{v['created_at'].strftime('%Y-%m-%d %H:%M')} ({v['size_mb']:.1f} MB)")
        total_size += v['size_mb']
    
    print(f"\nTotal space to be freed: {total_size:.1f} MB")
    
    if dry_run:
        print(f"\n{Colors.BLUE}DRY RUN - No files deleted{Colors.RESET}")
        return
    
    # Confirm deletion
    response = input(f"\n{Colors.YELLOW}Delete these {len(versions_to_delete)} versions? [y/N]: {Colors.RESET}")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete versions
    for v in versions_to_delete:
        try:
            delete_version_completely(v['version_id'], config, vm)
            print(f"{Colors.GREEN}✓{Colors.RESET} Deleted {v['version_id']}")
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.RESET} Error deleting {v['version_id']}: {e}")


def cleanup_failed_versions(dry_run: bool = False):
    """Delete all failed versions"""
    vm = VersionManager()
    config = get_config().config
    
    # Get failed versions
    failed_versions = vm.list_versions(status='failed', limit=None)
    
    if not failed_versions:
        print(f"{Colors.GREEN}No failed versions found.{Colors.RESET}")
        return
    
    # Display what will be deleted
    print(f"\n{Colors.YELLOW}Found {len(failed_versions)} failed versions:{Colors.RESET}")
    total_size = 0
    for v in failed_versions:
        version_id = v.get('version_id') or v.get('id')
        size = get_artifact_size(version_id, config)
        created_at = datetime.fromisoformat(v['created_at'])
        print(f"  • {version_id} - {created_at.strftime('%Y-%m-%d %H:%M')} ({size:.1f} MB)")
        total_size += size
    
    print(f"\nTotal space to be freed: {total_size:.1f} MB")
    
    if dry_run:
        print(f"\n{Colors.BLUE}DRY RUN - No files deleted{Colors.RESET}")
        return
    
    # Confirm deletion
    response = input(f"\n{Colors.YELLOW}Delete all {len(failed_versions)} failed versions? [y/N]: {Colors.RESET}")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete versions
    for v in failed_versions:
        try:
            delete_version_completely(v['version_id'], config, vm)
            print(f"{Colors.GREEN}✓{Colors.RESET} Deleted {v['version_id']}")
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.RESET} Error deleting {v['version_id']}: {e}")


def cleanup_by_pattern(pattern: str, dry_run: bool = False):
    """Delete versions matching a pattern"""
    vm = VersionManager()
    config = get_config().config
    
    # Get all versions
    all_versions = vm.list_versions(status='all', limit=None)
    versions_to_delete = []
    
    for version in all_versions:
        version_id = version.get('version_id') or version.get('id')
        if pattern.lower() in version_id.lower():
            size = get_artifact_size(version_id, config)
            versions_to_delete.append({
                'version_id': version_id,
                'status': version['status'],
                'created_at': datetime.fromisoformat(version['created_at']),
                'size_mb': size
            })
    
    if not versions_to_delete:
        print(f"{Colors.GREEN}No versions matching '{pattern}' found.{Colors.RESET}")
        return
    
    # Display what will be deleted
    print(f"\n{Colors.YELLOW}Found {len(versions_to_delete)} versions matching '{pattern}':{Colors.RESET}")
    total_size = 0
    for v in versions_to_delete:
        status_color = Colors.GREEN if v['status'] == 'completed' else Colors.RED
        print(f"  • {v['version_id']} - {status_color}{v['status']}{Colors.RESET} - "
              f"{v['created_at'].strftime('%Y-%m-%d %H:%M')} ({v['size_mb']:.1f} MB)")
        total_size += v['size_mb']
    
    print(f"\nTotal space to be freed: {total_size:.1f} MB")
    
    if dry_run:
        print(f"\n{Colors.BLUE}DRY RUN - No files deleted{Colors.RESET}")
        return
    
    # Confirm deletion
    response = input(f"\n{Colors.YELLOW}Delete these {len(versions_to_delete)} versions? [y/N]: {Colors.RESET}")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete versions
    for v in versions_to_delete:
        try:
            delete_version_completely(v['version_id'], config, vm)
            print(f"{Colors.GREEN}✓{Colors.RESET} Deleted {v['version_id']}")
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.RESET} Error deleting {v['version_id']}: {e}")


def keep_recent_versions(keep_count: int, dry_run: bool = False):
    """Keep only the N most recent successful versions"""
    vm = VersionManager()
    config = get_config().config
    
    # Get successful versions
    successful_versions = vm.list_versions(status='successful', limit=None)
    
    if len(successful_versions) <= keep_count:
        print(f"{Colors.GREEN}Only {len(successful_versions)} successful versions exist. "
              f"Nothing to delete (keeping {keep_count}).{Colors.RESET}")
        return
    
    # Sort by creation date (newest first)
    successful_versions.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Versions to keep and delete
    versions_to_keep = successful_versions[:keep_count]
    versions_to_delete = successful_versions[keep_count:]
    
    print(f"\n{Colors.GREEN}Keeping {keep_count} most recent successful versions:{Colors.RESET}")
    for v in versions_to_keep:
        created_at = datetime.fromisoformat(v['created_at'])
        print(f"  ✓ {v['version_id']} - {created_at.strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n{Colors.YELLOW}Deleting {len(versions_to_delete)} older successful versions:{Colors.RESET}")
    total_size = 0
    for v in versions_to_delete:
        size = get_artifact_size(v['version_id'], config)
        created_at = datetime.fromisoformat(v['created_at'])
        print(f"  • {v['version_id']} - {created_at.strftime('%Y-%m-%d %H:%M')} ({size:.1f} MB)")
        total_size += size
    
    print(f"\nTotal space to be freed: {total_size:.1f} MB")
    
    if dry_run:
        print(f"\n{Colors.BLUE}DRY RUN - No files deleted{Colors.RESET}")
        return
    
    # Confirm deletion
    response = input(f"\n{Colors.YELLOW}Delete these {len(versions_to_delete)} versions? [y/N]: {Colors.RESET}")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete versions
    for v in versions_to_delete:
        try:
            delete_version_completely(v['version_id'], config, vm)
            print(f"{Colors.GREEN}✓{Colors.RESET} Deleted {v['version_id']}")
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.RESET} Error deleting {v['version_id']}: {e}")


@click.group()
def cli():
    """Development version cleanup tools"""
    pass


@cli.command()
@click.option('--days', default=7, help='Delete versions older than N days')
@click.option('--include-successful', is_flag=True, help='Also delete successful versions')
@click.option('--dry-run', is_flag=True, help='Preview without deleting')
def age(days, include_successful, dry_run):
    """Delete versions older than specified days"""
    cleanup_by_age(days, dry_run, include_successful)


@cli.command()
@click.option('--dry-run', is_flag=True, help='Preview without deleting')
def failed(dry_run):
    """Delete all failed versions"""
    cleanup_failed_versions(dry_run)


@cli.command()
@click.argument('pattern')
@click.option('--dry-run', is_flag=True, help='Preview without deleting')
def pattern(pattern, dry_run):
    """Delete versions matching a pattern"""
    cleanup_by_pattern(pattern, dry_run)


@cli.command()
@click.option('--keep', default=5, help='Number of recent versions to keep')
@click.option('--dry-run', is_flag=True, help='Preview without deleting')
def keep_recent(keep, dry_run):
    """Keep only the N most recent successful versions"""
    keep_recent_versions(keep, dry_run)


@cli.command()
def interactive():
    """Interactive cleanup mode"""
    print(f"{Colors.BLUE}Interactive Development Cleanup{Colors.RESET}\n")
    
    vm = VersionManager()
    all_versions = vm.list_versions(status='all', limit=None)
    
    if not all_versions:
        print(f"{Colors.GREEN}No versions found. Environment is clean!{Colors.RESET}")
        return
    
    # Group versions by status
    successful = [v for v in all_versions if v['status'] == 'completed']
    failed = [v for v in all_versions if v['status'] != 'completed']
    
    print(f"Current versions:")
    print(f"  • {Colors.GREEN}{len(successful)} successful{Colors.RESET}")
    print(f"  • {Colors.RED}{len(failed)} failed/incomplete{Colors.RESET}")
    print(f"  • {len(all_versions)} total\n")
    
    print("Cleanup options:")
    print("1. Delete all failed versions")
    print("2. Delete versions older than N days")
    print("3. Keep only N most recent successful versions")
    print("4. Delete by pattern match")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == '1':
        cleanup_failed_versions()
    elif choice == '2':
        days = int(input("Delete versions older than how many days? "))
        include = input("Include successful versions? [y/N] ").lower() == 'y'
        cleanup_by_age(days, include_successful=include)
    elif choice == '3':
        keep = int(input("How many recent successful versions to keep? "))
        keep_recent_versions(keep)
    elif choice == '4':
        pattern = input("Enter pattern to match: ")
        cleanup_by_pattern(pattern)
    elif choice == '5':
        print("Exiting...")
    else:
        print(f"{Colors.RED}Invalid option{Colors.RESET}")


if __name__ == "__main__":
    cli()