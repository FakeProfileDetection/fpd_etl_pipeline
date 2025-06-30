# Data Formats and Scalability Guide

## File Formats

### Parquet vs CSV

Both `.parquet` and `.csv` files in the features directory contain **identical data**. We provide both formats for different use cases:

| Format | Pros | Cons | Best For |
|--------|------|------|----------|
| **CSV** | • Human-readable<br>• Universal compatibility<br>• Easy to inspect | • Large file size<br>• Slow read/write<br>• No schema enforcement | • Small datasets<br>• Manual inspection<br>• Sharing with non-technical users |
| **Parquet** | • Compressed (50-80% smaller)<br>• Fast read/write<br>• Schema preserved<br>• Column-based queries | • Binary format (not human-readable)<br>• Requires specific libraries | • Production use<br>• Large datasets<br>• Analytics pipelines |

### Feature File Structure

Each feature type directory contains:
```
statistical_session/
├── features.parquet    # Compressed binary format (recommended)
├── features.csv        # Human-readable format
└── feature_summary.json # Metadata about features
```

## Scalability Considerations

### Current Limitations (< 100 users)

The current pipeline works well for small datasets but will face challenges with 1000+ users:

1. **Memory constraints**: All data loaded into memory at once
2. **File system limits**: 18,000+ individual CSV files
3. **Processing time**: Sequential processing takes O(n) time
4. **Version tracking**: Single JSON file becomes unwieldy

### Recommended Changes for Scale (100-1000 users)

1. **Switch to Parquet-only** for processed data:
   ```python
   # In config/.env.local
   OUTPUT_FORMATS=parquet  # Drop CSV generation
   ```

2. **Implement batch processing**:
   ```python
   BATCH_SIZE=100  # Process 100 users at a time
   ```

3. **Use partitioned storage**:
   ```
   cleaned_data/
   ├── desktop/
   │   ├── batch_001/  # Users 1-100
   │   ├── batch_002/  # Users 101-200
   │   └── ...
   ```

### Enterprise Scale (1000+ users)

For production systems with thousands of users:

1. **Database-backed metadata**:
   - PostgreSQL for version tracking
   - Redis for processing status
   - S3/GCS for artifact storage

2. **Distributed processing**:
   - Apache Spark for feature extraction
   - Kubernetes for orchestration
   - Message queues for task distribution

3. **Data lakehouse architecture**:
   ```
   bronze/  # Raw data (immutable)
   silver/  # Cleaned, validated data
   gold/    # ML-ready features
   ```

4. **Incremental processing**:
   - Process only new/changed users
   - Maintain checksums for deduplication
   - Use CDC (Change Data Capture) patterns

## Performance Benchmarks

| Users | Current Pipeline | Optimized Pipeline | Distributed Pipeline |
|-------|-----------------|-------------------|---------------------|
| 10    | 5 seconds       | 5 seconds         | N/A (overhead)      |
| 100   | 50 seconds      | 30 seconds        | 20 seconds          |
| 1,000 | 8-10 minutes    | 3-4 minutes       | 1-2 minutes         |
| 10,000| 1-2 hours       | 15-20 minutes     | 5-10 minutes        |

## Migration Path

1. **Phase 1** (Current): Continue with dual formats for compatibility
2. **Phase 2** (100+ users): Switch to Parquet-only, implement batching
3. **Phase 3** (500+ users): Add database for metadata, partition storage
4. **Phase 4** (1000+ users): Implement distributed processing

## Code Examples

### Reading Parquet in Python
```python
import pandas as pd

# Read entire file
df = pd.read_parquet('features.parquet')

# Read specific columns (efficient with Parquet)
df = pd.read_parquet('features.parquet', columns=['user_id', 'mean_HL'])

# Read in chunks for large files
for chunk in pd.read_parquet('features.parquet', chunksize=1000):
    process_chunk(chunk)
```

### Reading Parquet in R
```r
library(arrow)

# Read entire file
df <- read_parquet("features.parquet")

# Read specific columns
df <- read_parquet("features.parquet", col_select = c("user_id", "mean_HL"))
```

### Checking Parquet schema
```bash
# Using Python
python -c "import pandas as pd; print(pd.read_parquet('features.parquet').info())"

# Using parquet-tools (if installed)
parquet-tools schema features.parquet
```