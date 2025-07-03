# Feature Extraction Documentation

## Overview

The FPD ETL Pipeline includes two types of feature extraction stages:

1. **Statistical Features** - Comprehensive statistical features for all keystroke patterns
2. **Top-K IL Features** - Focused subset of the most frequent Inter-key Latency features

## Directory Structure

```
artifacts/{version_id}/
├── statistical_features/           # All statistical features
│   ├── statistical_platform/       # Aggregated by user + platform
│   ├── statistical_session/        # Aggregated by user + platform + session
│   └── statistical_video/          # Aggregated by user + platform + session + video
└── statistical_IL_top_k_features/  # Top K IL features only
    ├── statistical_IL_top_10_platform/
    ├── statistical_IL_top_10_session/
    └── statistical_IL_top_10_video/
```

## Statistical Features

The main feature extraction stage (`extract_features`) generates comprehensive statistical features:

- **Unigram Features (HL)**: Hold Latency statistics for individual keys
  - Features: median, mean, std, q1, q3 for each key
  - Example: `HL_a_median`, `HL_a_mean`, etc.

- **Digram Features (IL)**: Inter-key Latency statistics for key pairs
  - Features: median, mean, std, q1, q3 for top 10 digrams
  - Example: `IL_av_median`, `IL_eKey.space_mean`, etc.

### Aggregation Levels

1. **Platform Level** (`statistical_platform`)
   - Groups data by user and platform
   - Highest level aggregation
   - Columns: user_id, platform_id, features...

2. **Session Level** (`statistical_session`)
   - Groups data by user, platform, and session
   - Medium granularity
   - Columns: user_id, platform_id, session_id, features...

3. **Video Level** (`statistical_video`)
   - Groups data by user, platform, session, and video
   - Most granular level
   - Columns: user_id, platform_id, session_id, video_id, features...

## Top-K IL Features

The `extract_top_il_features` stage creates focused datasets with only the most frequent IL features:

### Selection Strategy

1. Counts non-null occurrences of each IL feature across all records
2. Selects the top K features by frequency (default K=10)
3. Creates new datasets with only metadata columns + selected IL features

### Benefits

- **Reduced Dimensionality**: From ~400+ features to just 10-14 columns
- **Focus on Common Patterns**: Keeps only the most frequently observed digrams
- **Easier Analysis**: Simplified datasets for specific ML tasks

### Example Top Features

Common top IL features include:
- `IL_av_*` - Common digram "av"
- `IL_eKey.space_*` - Letter followed by space
- `IL_Key.spacet_*` - Space followed by letter
- `IL_th_*` - Common digram "th"
- `IL_ha_*` - Common digram "ha"

## Usage in Pipeline

```bash
# Run full pipeline including both feature extractions
python scripts/pipeline/run_pipeline.py --mode full

# Run only top-k IL feature extraction on existing data
python scripts/pipeline/extract_top_il_features.py --version-id <version_id> --k 10
```

## Feature File Formats

Both stages save features in two formats:
- `features.csv` - CSV format for compatibility
- `features.parquet` - Parquet format for efficiency
- `feature_summary.json` - Metadata about the features

## Imputation Strategies

- **Global Imputation**: Replace missing values with global mean (default)
- **User Imputation**: Replace with user-specific mean, fallback to global mean

The video-level features use user imputation by default for better personalization.