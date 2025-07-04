# Development Tools Guide

This guide covers specialized tools for managing versions and artifacts during development.

## ⚠️ WARNING: Development Only Tools

The cleanup and purge tools documented here are **ONLY** for use by solo developers during the development phase. They should **NEVER** be used in production or shared team environments.

## Version Management Tools

### 1. Version Tools CLI (`version_tools.py`)

The primary tool for version management, safe for all environments.

```bash
# List all versions
python scripts/standalone/version_tools.py list

# List with different formats
python scripts/standalone/version_tools.py list --format table  # Default, human-readable
python scripts/standalone/version_tools.py list --format json   # For processing
python scripts/standalone/version_tools.py list --format ids    # Just IDs for scripting

# Filter by status
python scripts/standalone/version_tools.py list --status successful
python scripts/standalone/version_tools.py list --status failed
python scripts/standalone/version_tools.py list --status archived

# Show detailed version information
python scripts/standalone/version_tools.py show 2025-07-03_10-58-43_loris-mbp

# Delete a specific version
python scripts/standalone/version_tools.py delete VERSION_ID
python scripts/standalone/version_tools.py delete VERSION_ID --artifacts  # Also delete artifacts

# Archive old versions
python scripts/standalone/version_tools.py archive VERSION_ID

# Clean up old failed versions
python scripts/standalone/version_tools.py cleanup --days 7
python scripts/standalone/version_tools.py cleanup --days 7 --dry-run

# Search for specific conditions
python scripts/standalone/version_tools.py search --stage-failed download_data
python scripts/standalone/version_tools.py search --days-ago 3

# Show statistics
python scripts/standalone/version_tools.py stats
```

### 2. Development Cleanup Tool (`cleanup_dev_versions.py`)

⚠️ **DEVELOPMENT ONLY** - Provides selective cleanup options

```bash
# Clean up failed versions only
python scripts/standalone/cleanup_dev_versions.py --failed-only

# Clean up versions older than N days
python scripts/standalone/cleanup_dev_versions.py --days-old 3

# Interactive mode - choose which to delete
python scripts/standalone/cleanup_dev_versions.py --interactive

# Keep specific number of recent versions
python scripts/standalone/cleanup_dev_versions.py --keep-recent 5

# Combine options
python scripts/standalone/cleanup_dev_versions.py --failed-only --days-old 1

# Always preview first!
python scripts/standalone/cleanup_dev_versions.py --dry-run
```

**Interactive Mode Example:**
```
Found 5 versions to potentially clean up:

1. [FAILED] 2025-07-03_09-15-22_loris-mbp (2 days old, 45.3 MB)
2. [FAILED] 2025-07-03_10-22-11_loris-mbp (1 day old, 12.1 MB)
3. [SUCCESS] 2025-07-03_11-33-44_loris-mbp (1 day old, 234.5 MB)
4. [FAILED] 2025-07-03_14-55-01_loris-mbp (5 hours old, 0.5 MB)
5. [SUCCESS] 2025-07-03_16-44-22_loris-mbp (2 hours old, 189.2 MB)

Select versions to delete (comma-separated numbers, 'all', or 'none'): 1,2,4
```

### 3. Nuclear Purge Tool (`purge_development_versions.py`)

🔥 **EXTREME CAUTION REQUIRED** 🔥

This tool permanently deletes ALL version data. It requires triple confirmation and should only be used when you want to completely reset your development environment.

```bash
# Preview what would be deleted (ALWAYS DO THIS FIRST)
python scripts/standalone/purge_development_versions.py --dry-run

# Purge all local version data (default)
python scripts/standalone/purge_development_versions.py

# Also delete cloud artifacts (opt-in)
python scripts/standalone/purge_development_versions.py --include-cloud

# Skip confirmation prompts (EXTREMELY DANGEROUS)
python scripts/standalone/purge_development_versions.py --force
```

**What gets deleted:**
- All version tracking files (versions_*.json)
- All individual version files (config/versions/*.json)
- All local artifacts (artifacts/*)
- Version history and metadata
- Cloud artifacts (only with --include-cloud flag)

**What is preserved:**
- Original web app data in cloud storage (/uploads/)
- Your code and configuration
- Test data in test_data/

**Triple Confirmation Process:**
1. Type "yes, delete everything"
2. Solve a math problem
3. Type a random confirmation code

## Version System Architecture

### File Structure
```
config/
├── versions_successful.json    # Tracks successful pipeline runs
├── versions_failed.json        # Tracks failed pipeline runs
├── versions_archived.json      # Old versions moved here
├── current_version.txt         # Points to active version
└── versions/                   # Detailed version information
    ├── 2025-07-03_10-58-43_loris-mbp.json
    ├── 2025-07-03_11-07-38_loris-mbp.json
    └── ...
```

### Git-Friendly Design

The system minimizes merge conflicts by:
1. **Separate files per version** - Each version creates its own JSON file
2. **Summary files only track metadata** - ID, status, timestamps
3. **Automatic cleanup** - Failed versions removed after 7 days
4. **Archive system** - Old successful versions moved to archive

### Team Collaboration

When multiple team members work simultaneously:
```bash
# Alice creates version: 2025-07-03_10-00-00_alice-laptop
# Bob creates version: 2025-07-03_10-05-00_bob-desktop

# Both push to git - no conflict!
# Different files created in config/versions/

# If summary file conflicts occur:
git pull
# Keep both entries in versions_successful.json
git add config/versions_successful.json
git commit -m "Merge: Keep both version entries"
git push
```

## Best Practices

### For Solo Development

1. **Regular Cleanup**
   ```bash
   # Weekly cleanup of failed versions
   python scripts/standalone/version_tools.py cleanup --days 7

   # Before presenting to team
   python scripts/standalone/cleanup_dev_versions.py --interactive
   ```

2. **Before Major Milestones**
   ```bash
   # Review what you have
   python scripts/standalone/version_tools.py stats

   # Clean up test runs
   python scripts/standalone/cleanup_dev_versions.py --failed-only

   # Upload good versions for team
   python scripts/standalone/upload_artifacts.py --version-id VERSION_ID
   ```

3. **Starting Fresh**
   ```bash
   # Only if you really need to start over
   python scripts/standalone/purge_development_versions.py --dry-run
   # Review carefully, then run without --dry-run if needed
   ```

### For Team Development

1. **Never use purge tools in shared environments**
2. **Commit version files after successful runs**
3. **Use cleanup tools only on your own failed versions**
4. **Coordinate before major cleanups**

## Automation with Version Tools

### Scripting Examples

```bash
# Delete all failed versions
for version in $(python scripts/standalone/version_tools.py list --status failed --format ids); do
    python scripts/standalone/version_tools.py delete $version --artifacts
done

# Archive versions older than 30 days
for version in $(python scripts/standalone/version_tools.py search --days-ago 30 --format ids); do
    python scripts/standalone/version_tools.py archive $version
done

# Get the latest successful version
LATEST=$(python scripts/standalone/version_tools.py list --status successful --limit 1 --format ids)
echo "Latest successful version: $LATEST"
```

## Safety Features

1. **Dry Run Mode** - All tools support --dry-run
2. **Confirmation Prompts** - Destructive operations require confirmation
3. **Cloud Opt-In** - Cloud deletion never happens by default
4. **Detailed Preview** - Shows exactly what will be deleted
5. **No Production Use** - Tools check for production markers

## Troubleshooting

### "Version not found"
- Check if version exists: `python scripts/standalone/version_tools.py list`
- Version might be archived: `python scripts/standalone/version_tools.py list --status archived`

### "Permission denied"
- Check file permissions: `ls -la config/versions*`
- Ensure you own the files: `chown -R $(whoami) config/`

### "Too many versions"
- Archive old versions: `python scripts/standalone/version_tools.py cleanup --days 30`
- Use cleanup tool: `python scripts/standalone/cleanup_dev_versions.py --keep-recent 10`

### Recovery
If you accidentally delete versions:
1. Check git history: `git log -- config/versions_successful.json`
2. Restore from git: `git checkout COMMIT -- config/versions*`
3. Download from cloud if uploaded: `python scripts/standalone/download_artifacts.py`
