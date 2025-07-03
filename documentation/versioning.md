1. Separated Version Tracking

  Instead of one massive versions.json file, we now have:
  - config/versions_successful.json - Only successful pipeline runs
  - config/versions_failed.json - Failed/incomplete runs
  - config/versions_archived/ - Old versions organized by year-month
  - config/versions/<version-id>.json - Individual files for each version

  2. Automatic Cleanup and Rotation

  - Failed versions are automatically cleaned up after 7 days (configurable)
  - Successful versions auto-archive when exceeding 50 recent versions
  - Archives are organized by year-month for easy browsing

  3. Git Conflict Resolution

  - Each version has its own JSON file in config/versions/
  - Main tracking files only contain summaries (ID, status, dates)
  - Team members can work independently without conflicts
  - current_version.txt tracks the active version

  4. Version Management CLI Tool

  New command: python scripts/standalone/version_tools.py

  List versions:
  # Show all versions
  python scripts/standalone/version_tools.py list

  # Show only failed versions
  python scripts/standalone/version_tools.py list --status failed

  # Show as JSON
  python scripts/standalone/version_tools.py list --format json

  Search versions:
  # Find versions where download_data failed
  python scripts/standalone/version_tools.py search --stage-failed download_data

  # Find versions from last 3 days
  python scripts/standalone/version_tools.py search --days-ago 3

  Delete versions:
  # Delete a specific version
  python scripts/standalone/version_tools.py delete 2025-06-29_21-23-04_hostname

  # Delete version and its artifacts
  python scripts/standalone/version_tools.py delete 2025-06-29_21-23-04_hostname --artifacts

  Clean up old failures:
  # Delete failed versions older than 7 days
  python scripts/standalone/version_tools.py cleanup --days 7

  # Preview what would be deleted
  python scripts/standalone/version_tools.py cleanup --days 7 --dry-run

  View statistics:
  python scripts/standalone/version_tools.py stats

  Output:
  VERSION STATISTICS
  ========================================
  Successful versions: 12
  Failed versions: 3
  Total versions: 15

  Current version: 2025-06-30_10-15-30_hostname
  Created: 2025-06-30T10:15:30

  Artifacts storage:
    Versions with artifacts: 8
    Total size: 234.5 MB

  Recent failures:
    - 2025-06-30_09-45-12_hostname... (download_data)
    - 2025-06-29_21-23-04_hostname... (run_eda)

  5. Developer Workflow

  For developers investigating failures:
  # Find all versions where download failed
  python scripts/standalone/version_tools.py search --stage-failed download_data

  # Show details of a specific version
  python scripts/standalone/version_tools.py show 2025-06-29_21-23-04_hostname

  # Export version info before deleting
  python scripts/standalone/version_tools.py show 2025-06-29_21-23-04_hostname --export version_info.json

  # Delete after fixing
  python scripts/standalone/version_tools.py delete 2025-06-29_21-23-04_hostname --artifacts

  6. Benefits

  - No more bloat: Failed versions auto-cleanup after 7 days
  - Easy navigation: Separate files for successful/failed runs
  - Team-friendly: Individual version files reduce merge conflicts
  - Scalable: Automatic archiving prevents unlimited growth
  - Developer-friendly: Powerful search and cleanup tools
  - Production-ready: File locking prevents corruption

  This system scales well from development (hundreds of test runs) to production (focused on successful versions only).

