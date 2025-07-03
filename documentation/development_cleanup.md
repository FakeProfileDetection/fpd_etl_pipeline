# Development Cleanup Tools

## Overview

During development, you'll create many test versions while building and testing the pipeline. These tools help manage and clean up development versions safely.

## Available Tools

### 1. Selective Cleanup Tool (`cleanup_dev_versions.py`)

A safer tool that allows selective deletion of versions based on various criteria.

```bash
# Delete all failed versions
python scripts/standalone/cleanup_dev_versions.py failed

# Delete versions older than 7 days (failed only)
python scripts/standalone/cleanup_dev_versions.py age --days 7

# Delete versions older than 7 days (including successful)
python scripts/standalone/cleanup_dev_versions.py age --days 7 --include-successful

# Keep only the 5 most recent successful versions
python scripts/standalone/cleanup_dev_versions.py keep-recent --keep 5

# Delete versions matching a pattern
python scripts/standalone/cleanup_dev_versions.py pattern "2025-07-03"

# Interactive mode - choose what to delete
python scripts/standalone/cleanup_dev_versions.py interactive

# Preview what would be deleted (dry run)
python scripts/standalone/cleanup_dev_versions.py failed --dry-run
```

### 2. Complete Purge Tool (`purge_development_versions.py`)

⚠️ **EXTREME CAUTION**: This tool deletes EVERYTHING - use only when you want a completely fresh start.

```bash
# Complete purge with triple confirmation
python scripts/standalone/purge_development_versions.py

# Preview what would be deleted
python scripts/standalone/purge_development_versions.py --dry-run

# Skip cloud deletion (local only)
python scripts/standalone/purge_development_versions.py --skip-cloud

# Force mode (skip confirmations - DANGEROUS!)
python scripts/standalone/purge_development_versions.py --force
```

## Safety Features

### Purge Tool Safety

1. **Prominent Warning Banner**: Large, colorful warning about what will be deleted
2. **Triple Confirmation Process**:
   - First: Type "yes, delete everything"
   - Second: Solve a simple math problem
   - Third: Type a random confirmation code
3. **5-Second Countdown**: Final chance to cancel (Ctrl+C)
4. **Detailed Preview**: Shows exactly what will be deleted before confirmation

### Cleanup Tool Safety

1. **Selective Deletion**: Only deletes what you specify
2. **Preview Mode**: All commands support `--dry-run`
3. **Confirmation Prompts**: Asks before deleting
4. **Status Awareness**: Won't delete successful versions unless explicitly requested

## Recommended Workflow

### During Development

1. **Regular Cleanup**: Use selective cleanup to remove failed runs
   ```bash
   python scripts/standalone/cleanup_dev_versions.py failed
   ```

2. **Keep Recent Work**: Maintain only recent successful versions
   ```bash
   python scripts/standalone/cleanup_dev_versions.py keep-recent --keep 3
   ```

3. **Pattern-Based Cleanup**: Remove specific day's experiments
   ```bash
   python scripts/standalone/cleanup_dev_versions.py pattern "2025-07-03"
   ```

### Before Team Presentation

1. **Complete Fresh Start**: Use purge tool for a clean slate
   ```bash
   # First, do a dry run to see what will be deleted
   python scripts/standalone/purge_development_versions.py --dry-run
   
   # If satisfied, run the actual purge
   python scripts/standalone/purge_development_versions.py
   ```

2. **Run Final Pipeline**: Create your presentation-ready version
   ```bash
   python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts
   ```

## What Gets Deleted

### Local Files
- Version tracking files (versions_*.json)
- Individual version JSON files
- Artifact directories
- Logs and metadata

### Cloud Storage
- All uploaded artifacts
- Version-specific data in GCS

## Recovery

**IMPORTANT**: These deletions are permanent. There is no undo.

Before using these tools:
- Ensure you don't need any of the data
- Consider backing up important results
- Verify you're in the right environment

## Best Practices

1. **Always use dry-run first**: See what will be deleted
2. **Start with selective cleanup**: Only use purge when necessary
3. **Document important findings**: Before deleting versions
4. **Coordinate with team**: Ensure no one needs the data
5. **Fresh start protocol**: Purge → Run pipeline → Upload → Present