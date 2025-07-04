#!/usr/bin/env python3
"""
List Available Pipeline Versions
Shows all versions with their status and metadata
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config_manager import get_config
from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_timestamp


@click.command()
@click.option("--limit", default=10, help="Number of versions to show (default: 10)")
@click.option("--all", "show_all", is_flag=True, help="Show all versions")
@click.option("--complete-only", is_flag=True, help="Show only completed versions")
@click.option(
    "--uploaded-only", is_flag=True, help="Show only versions with uploaded artifacts"
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def main(
    limit: int,
    show_all: bool,
    complete_only: bool,
    uploaded_only: bool,
    output_json: bool,
):
    """List available pipeline versions with their status"""

    # Load version manager
    vm = VersionManager()
    versions = vm.list_versions()

    if not versions:
        click.echo("No versions found")
        return

    # Filter versions
    filtered_versions = []
    for v in versions:
        if complete_only and v.get("status") != "complete":
            continue
        if uploaded_only and not v.get("summary", {}).get("artifacts_uploaded"):
            continue
        filtered_versions.append(v)

    # Sort by creation time (newest first)
    filtered_versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Limit results
    if not show_all:
        filtered_versions = filtered_versions[:limit]

    # Output as JSON if requested
    if output_json:
        click.echo(json.dumps(filtered_versions, indent=2))
        return

    # Load cloud manager to check uploaded artifacts
    config = get_config()
    if config.validate_cloud_config():
        # CloudArtifactManager needs version_id, but we're listing versions
        # so we'll skip cloud checking for now
        pass

    # Display results
    click.echo("\n📋 Pipeline Versions")
    click.echo("=" * 80)

    # Find and highlight the latest complete version with artifacts
    latest_complete_uploaded = None
    for v in filtered_versions:
        if (
            v.get("status") == "complete"
            and v.get("summary", {}).get("artifacts_uploaded")
            and not latest_complete_uploaded
        ):
            latest_complete_uploaded = v.get("id") or v.get("version_id")

    for i, version in enumerate(filtered_versions):
        version_id = version.get("id") or version.get("version_id")
        created = format_timestamp(version.get("created_at", "Unknown"))
        status = version.get("status", "unknown")
        user = version.get("user", "unknown")

        # Status emoji
        if status == "complete":
            status_emoji = "✅"
        elif status == "in_progress":
            status_emoji = "🔄"
        else:
            status_emoji = "❌"

        # Check stages
        stages = version.get("stages", {})
        completed_stages = [
            name for name, info in stages.items() if info.get("completed")
        ]

        # Check if artifacts uploaded
        uploaded = version.get("summary", {}).get("artifacts_uploaded", False)
        upload_emoji = "☁️" if uploaded else "💾"

        # Highlight latest complete uploaded version
        if version_id == latest_complete_uploaded:
            click.echo("\n⭐ LATEST COMPLETE VERSION WITH ARTIFACTS:")
        else:
            click.echo(f"\n{i+1}.")

        click.echo(f"   Version: {version_id}")
        click.echo(f"   Created: {created} by {user}")
        click.echo(f"   Status: {status_emoji} {status} {upload_emoji}")

        if completed_stages:
            click.echo(f"   Stages: {', '.join(completed_stages)}")

        # Show summary stats if available
        if "summary" in version and status == "complete":
            summary = version["summary"]
            if "stages_run" in summary:
                click.echo(f"   Completed: {len(summary['stages_run'])} stages")

        # Check cloud artifacts if uploaded
        # TODO: Fix cloud manager initialization for listing
        # if uploaded and cloud_manager:
        #     try:
        #         manifest = cloud_manager.get_artifact_manifest(version_id)
        #         if manifest:
        #             click.echo(f"   Cloud: {manifest.get('total_files', 0)} files uploaded")
        #     except:
        #         pass

    # Show usage tips
    click.echo("\n💡 Tips:")
    if latest_complete_uploaded:
        click.echo(
            f"   Download latest: python scripts/standalone/download_artifacts.py --version-id {latest_complete_uploaded}"
        )
    click.echo("   Show all versions: python scripts/standalone/list_versions.py --all")
    click.echo(
        "   Show only uploaded: python scripts/standalone/list_versions.py --uploaded-only"
    )


if __name__ == "__main__":
    main()
