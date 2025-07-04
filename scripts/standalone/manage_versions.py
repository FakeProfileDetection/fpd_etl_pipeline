#!/usr/bin/env python3
"""
Manage Pipeline Versions
Clean up old versions and migrate to new system
"""

import sys
import click
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)


@click.group()
def cli():
    """Manage pipeline versions"""
    pass


@cli.command()
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deleted without deleting"
)
@click.option("--keep-count", default=10, help="Number of local versions to keep")
@click.option(
    "--include-uploaded", is_flag=True, help="Also clean uploaded versions (dangerous!)"
)
def cleanup(dry_run: bool, keep_count: int, include_uploaded: bool):
    """Clean up old versions and artifacts"""

    # Use V1 for now until migration
    vm = VersionManager()
    versions = vm.list_versions()

    click.echo(f"Found {len(versions)} total versions")

    # Categorize versions
    test_versions = []
    complete_uploaded = []
    complete_local = []
    in_progress = []

    for v in versions:
        version_id = v["version_id"]

        # Skip if it's a test version
        if "test" in version_id or v.get("metadata", {}).get("test"):
            test_versions.append(v)
        elif v.get("status") == "complete":
            if v.get("summary", {}).get("artifacts_uploaded"):
                complete_uploaded.append(v)
            else:
                complete_local.append(v)
        else:
            in_progress.append(v)

    click.echo("\nVersion breakdown:")
    click.echo(f"  Test versions: {len(test_versions)}")
    click.echo(f"  Complete (uploaded): {len(complete_uploaded)}")
    click.echo(f"  Complete (local): {len(complete_local)}")
    click.echo(f"  In progress: {len(in_progress)}")

    to_delete = []

    # Always delete test versions
    to_delete.extend(test_versions)

    # Delete old in-progress versions (> 7 days)
    for v in in_progress:
        created = datetime.fromisoformat(v["created_at"].replace("Z", "+00:00"))
        age_days = (datetime.now() - created).days
        if age_days > 7:
            to_delete.append(v)

    # Keep only recent local versions
    if len(complete_local) > keep_count:
        # Sort by date, keep newest
        complete_local.sort(key=lambda x: x["created_at"], reverse=True)
        to_delete.extend(complete_local[keep_count:])

    # Optionally delete uploaded versions
    if include_uploaded and len(complete_uploaded) > keep_count:
        complete_uploaded.sort(key=lambda x: x["created_at"], reverse=True)
        to_delete.extend(complete_uploaded[keep_count:])

    if not to_delete:
        click.echo("\nNo versions to delete")
        return

    click.echo(f"\nVersions to delete: {len(to_delete)}")
    for v in to_delete[:10]:  # Show first 10
        age_days = (
            datetime.now()
            - datetime.fromisoformat(v["created_at"].replace("Z", "+00:00"))
        ).days
        click.echo(f"  - {v['version_id']} ({age_days} days old)")

    if len(to_delete) > 10:
        click.echo(f"  ... and {len(to_delete) - 10} more")

    # Calculate space savings
    total_size = 0
    for v in to_delete:
        artifacts_dir = Path("artifacts") / v["version_id"]
        if artifacts_dir.exists():
            size = sum(
                f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()
            )
            total_size += size

    click.echo(f"\nEstimated space savings: {total_size / (1024**3):.2f} GB")

    if dry_run:
        click.echo("\n🔍 DRY RUN - No files deleted")
        return

    if not click.confirm("\nProceed with deletion?"):
        click.echo("Cancelled")
        return

    # Delete versions
    deleted_count = 0
    for v in to_delete:
        version_id = v["version_id"]

        # Delete artifacts directory
        artifacts_dir = Path("artifacts") / version_id
        if artifacts_dir.exists():
            import shutil

            shutil.rmtree(artifacts_dir)

        deleted_count += 1

    click.echo(f"\n✅ Deleted {deleted_count} versions")

    # Update versions.json
    remaining_versions = [v for v in versions if v not in to_delete]
    vm.data["versions"] = remaining_versions

    # Update current if needed
    if vm.data["current"] in [v["version_id"] for v in to_delete]:
        vm.data["current"] = (
            remaining_versions[0]["version_id"] if remaining_versions else None
        )

    # Save updated file
    with open(vm.versions_file, "w") as f:
        json.dump(vm.data, f, indent=2)

    click.echo("Updated versions.json")


@cli.command()
@click.option("--force", is_flag=True, help="Force migration even if already migrated")
def migrate(force: bool):
    """Migrate from versions.json to directory structure"""

    versions_dir = Path("versions")
    if versions_dir.exists() and not force:
        click.echo("❌ Versions directory already exists. Use --force to re-migrate.")
        return

    old_file = Path("versions.json")
    if not old_file.exists():
        click.echo("❌ No versions.json found")
        return

    click.echo("🔄 Migrating to directory-based version system...")

    try:
        migrate_from_v1()
        click.echo("✅ Migration complete!")
        click.echo("\nNext steps:")
        click.echo("1. Update code to use VersionManagerV2")
        click.echo("2. Test with a few pipeline runs")
        click.echo("3. Delete the backup file when confident")
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()


@cli.command()
@click.argument("version_id")
def show(version_id: str):
    """Show details for a specific version"""

    # Try V2 first
    if Path("versions").exists():
        vm = VersionManagerV2()
        version_data = vm.get_version(version_id)
    else:
        # Fall back to V1
        vm = VersionManager()
        version_data = vm.get_version(version_id)

    if not version_data:
        click.echo(f"❌ Version {version_id} not found")
        return

    # Pretty print
    click.echo(json.dumps(version_data, indent=2))

    # Show artifacts info
    artifacts_dir = Path("artifacts") / version_id
    if artifacts_dir.exists():
        total_files = sum(1 for _ in artifacts_dir.rglob("*") if _.is_file())
        total_size = sum(
            f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()
        )
        click.echo(f"\nArtifacts: {total_files} files, {total_size / (1024**2):.2f} MB")


@cli.command()
def stats():
    """Show version statistics"""

    # Use appropriate manager
    if Path("versions").exists():
        vm = VersionManagerV2()
        versions = vm.list_versions(limit=1000)
    else:
        vm = VersionManager()
        versions = vm.list_versions()

    # Calculate stats
    total = len(versions)
    complete = sum(1 for v in versions if v.get("status") == "complete")
    uploaded = sum(
        1 for v in versions if v.get("summary", {}).get("artifacts_uploaded")
    )

    # Group by user
    by_user = {}
    for v in versions:
        user = v.get("user", "unknown")
        by_user[user] = by_user.get(user, 0) + 1

    # Group by date
    by_date = {}
    for v in versions:
        date = v["created_at"][:10]  # YYYY-MM-DD
        by_date[date] = by_date.get(date, 0) + 1

    click.echo("📊 Version Statistics")
    click.echo("=" * 40)
    click.echo(f"Total versions: {total}")
    click.echo(f"Complete: {complete}")
    click.echo(f"Uploaded: {uploaded}")
    click.echo(f"In progress: {total - complete}")

    click.echo("\nBy user:")
    for user, count in sorted(by_user.items(), key=lambda x: x[1], reverse=True):
        click.echo(f"  {user}: {count}")

    click.echo("\nBy date (last 7 days):")
    dates = sorted(by_date.keys(), reverse=True)[:7]
    for date in dates:
        click.echo(f"  {date}: {by_date[date]}")

    # Disk usage
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        total_size = sum(
            f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()
        )
        click.echo(f"\nTotal artifacts size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    cli()
