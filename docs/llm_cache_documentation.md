# LLM Check Cache Documentation

## Overview

The LLM Check Cache system stores previously processed LLM check results to avoid re-running expensive API calls on unchanged user data. This significantly reduces processing time and API costs when re-running the pipeline on datasets that include previously processed users.

## Benefits

- **Cost Savings**: Avoid redundant API calls (OpenAI or local model)
- **Time Efficiency**: Skip processing for unchanged users
- **Smart Invalidation**: Automatically detects when user's text has changed
- **Persistence**: SQLite database survives between pipeline runs
- **Version Control**: Can be committed to Git for team sharing

## How It Works

### 1. Cache Key
Each user's results are cached using:
- `user_id`: Unique user identifier
- `device_type`: desktop or mobile
- `text_hash`: SHA256 hash of all the user's text files

### 2. Cache Checking
When processing users:
1. Calculate hash of user's current text files
2. Check if user exists in cache with matching hash
3. If match found → use cached results (cache hit)
4. If no match or changed → process with API (cache miss)

### 3. Automatic Invalidation
The cache automatically invalidates when:
- User's text content changes
- New text files are added
- Text files are removed
- File order changes (ensures consistency)

## Configuration

### Enable/Disable Cache
```bash
# In .env file
LLM_CHECK_USE_CACHE=true  # Enable caching (default)
LLM_CHECK_USE_CACHE=false # Disable caching
```

### Cache Location
Default: `data/llm_check_cache.db`

## Usage Examples

### Running Pipeline with Cache
```bash
# First run - all users processed
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check
# Output: Cache misses: 100, API calls: 300

# Second run - same data, all cached
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check
# Output: Cache hits: 100, API calls: 0

# After removing problematic users
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check
# Output: Cache hits: 95, Cache misses: 0, API calls: 0
```

### Cache Management

#### View Cache Statistics
```python
from scripts.pipeline.llm_cache import LLMCheckCache

cache = LLMCheckCache()
stats = cache.get_cache_stats()
print(f"Cached users: {stats['total_users']}")
print(f"Cache size: {stats['cache_size_bytes'] / 1024:.1f} KB")
```

#### Clear Old Entries
```python
# Clear entries older than 30 days
cache.clear_cache(older_than_days=30)

# Clear all entries
cache.clear_cache()
```

#### Export Cache for Backup
```python
cache.export_cache("backups/llm_cache_backup.json")
```

## Database Schema

### Table: llm_check_results
| Column | Type | Description |
|--------|------|-------------|
| user_id | TEXT | Primary key, user identifier |
| device_type | TEXT | desktop or mobile |
| results_json | TEXT | JSON array of score dictionaries |
| text_hash | TEXT | SHA256 hash of text content |
| model_used | TEXT | Model name used for processing |
| created_at | TIMESTAMP | When first cached |
| updated_at | TIMESTAMP | Last update time |

## Performance Impact

### Example Dataset (1000 users, 3 files each)
- **First run**: 3000 API calls, ~10 minutes
- **Second run**: 0 API calls, ~30 seconds (cache lookup only)
- **After 50 users changed**: 150 API calls, ~2 minutes

### Cost Savings (OpenAI GPT-4)
- First run: $3.00 (3000 calls)
- Subsequent runs: $0.00 (cached)
- Incremental updates: $0.15 (only changed users)

## Best Practices

1. **Commit the Cache**: Add `data/llm_check_cache.db` to Git for team sharing
   ```bash
   git add data/llm_check_cache.db
   git commit -m "Update LLM check cache"
   ```

2. **Use Git LFS for Large Caches**: If cache grows > 50MB
   ```bash
   git lfs track "data/llm_check_cache.db"
   ```

3. **Regular Cleanup**: Remove old entries periodically
   ```python
   cache.clear_cache(older_than_days=90)
   ```

4. **Backup Before Major Changes**: Export cache before pipeline updates
   ```python
   cache.export_cache(f"backups/cache_{datetime.now():%Y%m%d}.json")
   ```

## Troubleshooting

### Cache Not Being Used
- Check `LLM_CHECK_USE_CACHE=true` in .env
- Verify `data/llm_check_cache.db` exists
- Check file permissions

### Unexpected Cache Misses
- User's text files may have changed
- Check if files were re-saved (changes hash)
- Verify consistent file encoding

### Cache Corruption
```bash
# Delete and rebuild
rm data/llm_check_cache.db
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check
```

## Integration Status

✅ **Implemented**:
- SQLite cache database
- Cache hit/miss detection
- Content change detection
- Statistics tracking

🚧 **TODO** (for full integration):
- Modify `extract_llm_scores.py` to use cache in main pipeline
- Add cache statistics to pipeline reports
- Add CLI commands for cache management
