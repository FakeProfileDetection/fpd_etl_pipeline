#!/usr/bin/env python3
"""
DEVELOPMENT VERSION PURGE TOOL
⚠️  EXTREME CAUTION: This tool permanently deletes ALL version data! ⚠️

This tool is intended ONLY for solo developers during the development phase.
It will DELETE ALL:
- Local version tracking files
- Local artifact directories
- Cloud storage artifacts
- Version history

DO NOT USE IN PRODUCTION OR SHARED ENVIRONMENTS!
"""

import click
import sys
import shutil
import time
import random
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import EnhancedVersionManager as VersionManager
from scripts.utils.config_manager import get_config
from scripts.utils.cloud_artifact_manager import CloudArtifactManager


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    RESET = '\033[0m'


def print_warning_banner():
    """Display a prominent warning banner"""
    banner = f"""
{Colors.RED}{Colors.BOLD}{'='*80}
{Colors.BLINK}⚠️  EXTREME DANGER ZONE ⚠️{Colors.RESET}{Colors.RED}{Colors.BOLD}
{'='*80}{Colors.RESET}

{Colors.YELLOW}{Colors.BOLD}YOU ARE ABOUT TO PERMANENTLY DELETE ALL VERSION DATA!{Colors.RESET}

{Colors.RED}This tool will IRREVERSIBLY delete:{Colors.RESET}
  • ALL local version tracking files
  • ALL local artifact directories  
  • ALL version history and metadata
  • ALL pipeline outputs and logs
  • Cloud artifacts (ONLY if --include-cloud is used)

{Colors.MAGENTA}{Colors.BOLD}This action CANNOT be undone!{Colors.RESET}

{Colors.YELLOW}{Colors.UNDERLINE}This tool should ONLY be used:{Colors.RESET}
  ✓ By solo developers during development
  ✓ When you want to clean up test/development runs
  ✓ BEFORE presenting to your team
  
{Colors.RED}{Colors.BOLD}This tool should NEVER be used:{Colors.RESET}
  ✗ In production environments
  ✗ When working with a team
  ✗ When data has any value
  ✗ Without understanding the consequences

{Colors.RED}{Colors.BOLD}{'='*80}{Colors.RESET}
"""
    print(banner)


def print_items_to_delete(local_versions: List[str], cloud_versions: List[str], 
                         artifacts_dir: Path, config_dir: Path):
    """Display what will be deleted"""
    print(f"\n{Colors.YELLOW}The following will be PERMANENTLY deleted:{Colors.RESET}\n")
    
    print(f"{Colors.BLUE}Local Version Files:{Colors.RESET}")
    version_files = [
        config_dir / "versions_successful.json",
        config_dir / "versions_failed.json", 
        config_dir / "versions_archived.json",
        config_dir / "current_version.txt"
    ]
    for vf in version_files:
        if vf.exists():
            print(f"  • {vf}")
    
    versions_dir = config_dir / "versions"
    if versions_dir.exists():
        version_count = len(list(versions_dir.glob("*.json")))
        print(f"  • {versions_dir}/ ({version_count} individual version files)")
    
    print(f"\n{Colors.BLUE}Local Artifacts:{Colors.RESET}")
    if artifacts_dir.exists():
        local_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
        print(f"  • {len(local_dirs)} version directories in {artifacts_dir}/")
        if len(local_dirs) <= 10:
            for d in local_dirs:
                size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                print(f"    - {d.name} ({size/1024/1024:.1f} MB)")
        else:
            print(f"    - (showing first 5 of {len(local_dirs)})")
            for d in local_dirs[:5]:
                size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                print(f"    - {d.name} ({size/1024/1024:.1f} MB)")
            print(f"    - ... and {len(local_dirs) - 5} more")
    
    print(f"\n{Colors.BLUE}Cloud Storage:{Colors.RESET}")
    if cloud_versions:
        print(f"  • {len(cloud_versions)} versions in cloud storage")
        if len(cloud_versions) <= 10:
            for v in cloud_versions:
                print(f"    - {v}")
        else:
            print(f"    - (showing first 5 of {len(cloud_versions)})")
            for v in cloud_versions[:5]:
                print(f"    - {v}")
            print(f"    - ... and {len(cloud_versions) - 5} more")
    else:
        print("  • No cloud versions found (or cloud storage not configured)")


def get_random_confirmation_code() -> str:
    """Generate a random confirmation code"""
    words = ["DELETE", "PURGE", "DESTROY", "REMOVE", "ERASE", "WIPE"]
    numbers = random.randint(1000, 9999)
    return f"{random.choice(words)}-{numbers}"


def triple_confirmation() -> bool:
    """Triple confirmation process with increasing difficulty"""
    
    # First confirmation
    print(f"\n{Colors.YELLOW}{Colors.BOLD}FIRST CONFIRMATION:{Colors.RESET}")
    print(f"{Colors.RED}Are you ABSOLUTELY SURE you want to delete ALL version data?{Colors.RESET}")
    response = input(f"Type {Colors.BOLD}'yes, delete everything'{Colors.RESET} to continue: ")
    if response.lower() != "yes, delete everything":
        print(f"{Colors.GREEN}Purge cancelled. Good choice!{Colors.RESET}")
        return False
    
    # Second confirmation with math
    print(f"\n{Colors.YELLOW}{Colors.BOLD}SECOND CONFIRMATION:{Colors.RESET}")
    num1 = random.randint(10, 50)
    num2 = random.randint(10, 50)
    answer = num1 + num2
    print(f"{Colors.RED}This action is IRREVERSIBLE. Prove you're paying attention.{Colors.RESET}")
    response = input(f"What is {num1} + {num2}? ")
    try:
        if int(response) != answer:
            print(f"{Colors.RED}Incorrect answer. Purge cancelled for safety.{Colors.RESET}")
            return False
    except ValueError:
        print(f"{Colors.RED}Invalid input. Purge cancelled for safety.{Colors.RESET}")
        return False
    
    # Third confirmation with random code
    print(f"\n{Colors.YELLOW}{Colors.BOLD}FINAL CONFIRMATION:{Colors.RESET}")
    code = get_random_confirmation_code()
    print(f"{Colors.RED}{Colors.BLINK}THIS IS YOUR LAST CHANCE TO CANCEL!{Colors.RESET}")
    print(f"{Colors.RED}You are about to PERMANENTLY DELETE all version data.{Colors.RESET}")
    print(f"\nTo proceed, type this confirmation code: {Colors.BOLD}{Colors.UNDERLINE}{code}{Colors.RESET}")
    response = input("Confirmation code: ")
    if response != code:
        print(f"{Colors.GREEN}Confirmation code incorrect. Purge cancelled. Phew!{Colors.RESET}")
        return False
    
    # Final countdown
    print(f"\n{Colors.RED}{Colors.BOLD}Initiating purge in:{Colors.RESET}")
    for i in range(5, 0, -1):
        print(f"{Colors.RED}{Colors.BOLD}{i}...{Colors.RESET}", end='', flush=True)
        time.sleep(1)
    print(f"\n{Colors.RED}{Colors.BOLD}PURGING ALL DATA...{Colors.RESET}\n")
    
    return True


def delete_local_versions(config_dir: Path, dry_run: bool = False):
    """Delete all local version tracking files"""
    print(f"{Colors.YELLOW}Deleting local version files...{Colors.RESET}")
    
    # Version tracking files
    version_files = [
        "versions_successful.json",
        "versions_failed.json",
        "versions_archived.json",
        "current_version.txt"
    ]
    
    for vf in version_files:
        file_path = config_dir / vf
        if file_path.exists():
            if not dry_run:
                file_path.unlink()
            print(f"  {'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {vf}")
    
    # Individual version files
    versions_dir = config_dir / "versions"
    if versions_dir.exists():
        version_files = list(versions_dir.glob("*.json"))
        if not dry_run:
            shutil.rmtree(versions_dir)
        print(f"  {'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {len(version_files)} individual version files")


def delete_local_artifacts(artifacts_dir: Path, dry_run: bool = False):
    """Delete all local artifact directories"""
    print(f"\n{Colors.YELLOW}Deleting local artifacts...{Colors.RESET}")
    
    if not artifacts_dir.exists():
        print("  No artifacts directory found")
        return
    
    version_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    total_size = 0
    
    for vdir in version_dirs:
        dir_size = sum(f.stat().st_size for f in vdir.rglob('*') if f.is_file())
        total_size += dir_size
        if not dry_run:
            shutil.rmtree(vdir)
        print(f"  {'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {vdir.name} ({dir_size/1024/1024:.1f} MB)")
    
    print(f"\n  Total space {'to be freed' if dry_run else 'freed'}: {total_size/1024/1024:.1f} MB")


def delete_cloud_artifacts(cloud_versions: List[str], config: Dict[str, Any], dry_run: bool = False):
    """Delete all cloud storage artifacts"""
    print(f"\n{Colors.YELLOW}Deleting cloud artifacts...{Colors.RESET}")
    
    if not cloud_versions:
        print("  No cloud versions to delete")
        return
    
    bucket_name = config.get("GCS_BUCKET_NAME")
    if not bucket_name:
        print("  Cloud storage not configured")
        return
    
    for version in cloud_versions:
        try:
            # Delete the entire version folder
            cmd = ["gsutil", "-m", "rm", "-r", f"gs://{bucket_name}/artifacts/{version}/"]
            if not dry_run:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  {Colors.RED}Error deleting {version}: {result.stderr}{Colors.RESET}")
                else:
                    print(f"  Deleted cloud artifacts: {version}")
            else:
                print(f"  [DRY RUN] Would delete cloud artifacts: {version}")
        except Exception as e:
            print(f"  {Colors.RED}Error deleting {version}: {e}{Colors.RESET}")


def get_cloud_versions(config: Dict[str, Any]) -> List[str]:
    """Get list of versions in cloud storage"""
    bucket_name = config.get("GCS_BUCKET_NAME")
    if not bucket_name:
        return []
    
    try:
        # List all version directories in artifacts/
        cmd = ["gsutil", "ls", f"gs://{bucket_name}/artifacts/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return []
        
        # Parse version IDs from the output
        versions = []
        for line in result.stdout.strip().split('\n'):
            if line.endswith('/'):
                # Extract version ID from path like gs://bucket/artifacts/version-id/
                version = line.split('/')[-2]
                if version:
                    versions.append(version)
        
        return versions
    except Exception:
        return []


def create_fresh_version_files(config_dir: Path):
    """Create fresh, empty version tracking files"""
    print(f"\n{Colors.GREEN}Creating fresh version tracking files...{Colors.RESET}")
    
    # Create empty version files
    empty_versions = {
        "versions": {},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    }
    
    for filename in ["versions_successful.json", "versions_failed.json", "versions_archived.json"]:
        file_path = config_dir / filename
        with open(file_path, 'w') as f:
            json.dump(empty_versions, f, indent=2)
        print(f"  Created: {filename}")
    
    # Create versions directory
    versions_dir = config_dir / "versions"
    versions_dir.mkdir(exist_ok=True)
    print(f"  Created: versions/ directory")
    
    # Remove current_version.txt (will be created on next run)
    current_version_file = config_dir / "current_version.txt"
    if current_version_file.exists():
        current_version_file.unlink()


@click.command()
@click.option('--dry-run', is_flag=True, help='Preview what would be deleted without actually deleting')
@click.option('--include-cloud', is_flag=True, help='Also delete cloud artifacts (default: local only)')
@click.option('--force', is_flag=True, help='Skip confirmation prompts (DANGEROUS!)')
def purge_all_versions(dry_run, include_cloud, force):
    """
    Purge ALL version data - local only by default
    
    ⚠️  EXTREME CAUTION: This permanently deletes ALL version history! ⚠️
    
    By default, only local data is deleted. Use --include-cloud to also delete cloud artifacts.
    """
    # Display warning banner
    print_warning_banner()
    
    # Get configuration
    config = get_config().config
    artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts"))
    config_dir = Path("config")
    
    # Get list of versions
    vm = VersionManager()
    local_versions = []
    try:
        all_versions = vm.list_versions(status='all', limit=None)
        local_versions = [v['version_id'] for v in all_versions]
    except Exception:
        # If version manager fails, scan artifacts directory
        if artifacts_dir.exists():
            local_versions = [d.name for d in artifacts_dir.iterdir() if d.is_dir()]
    
    # Get cloud versions
    cloud_versions = []
    if include_cloud:
        print(f"{Colors.BLUE}Checking cloud storage...{Colors.RESET}")
        cloud_versions = get_cloud_versions(config)
    
    # Show what will be deleted
    print_items_to_delete(local_versions, cloud_versions, artifacts_dir, config_dir)
    
    # Check if there's anything to delete
    if not local_versions and not cloud_versions and not artifacts_dir.exists():
        print(f"\n{Colors.GREEN}No versions found to delete. Environment is already clean!{Colors.RESET}")
        return
    
    # Confirmation process
    if not force:
        if not triple_confirmation():
            return
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}--force flag used. Skipping confirmations!{Colors.RESET}")
        if not dry_run:
            time.sleep(2)  # Brief pause for safety
    
    # Execute purge
    if dry_run:
        print(f"\n{Colors.BLUE}{Colors.BOLD}DRY RUN MODE - No actual deletions{Colors.RESET}\n")
    
    try:
        # Delete local version files
        delete_local_versions(config_dir, dry_run)
        
        # Delete local artifacts
        delete_local_artifacts(artifacts_dir, dry_run)
        
        # Delete cloud artifacts
        if include_cloud and cloud_versions:
            delete_cloud_artifacts(cloud_versions, config, dry_run)
        
        # Create fresh version files
        if not dry_run:
            create_fresh_version_files(config_dir)
        
        # Success message
        if dry_run:
            print(f"\n{Colors.BLUE}{Colors.BOLD}DRY RUN COMPLETE{Colors.RESET}")
            print("No files were actually deleted. Remove --dry-run to execute.")
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ PURGE COMPLETE!{Colors.RESET}")
            print(f"{Colors.GREEN}All version data has been permanently deleted.{Colors.RESET}")
            print(f"{Colors.GREEN}Fresh version tracking files have been created.{Colors.RESET}")
            print(f"\n{Colors.YELLOW}You can now start fresh with:{Colors.RESET}")
            print(f"  python scripts/pipeline/run_pipeline.py --mode full")
            
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}ERROR during purge: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    purge_all_versions()