# Data Schema Documentation

## File Naming Conventions

### Raw Keystroke Files
- **Web app format**: `{platform}_{user_id}_{sequence}.csv`
  - Example: `f_a1b2c3d4e5f6789012345678901234ef_0.csv`
  - platform: f (Facebook), i (Instagram), t (Twitter)
  - user_id: 32-character hex string (MD5 hash)
  - sequence: 0-17 (see sequence mapping below)

- **TypeNet format** (after clean_data): `{platform_id}_{video_id}_{session_id}_{user_id}.csv`
  - Example: `1_1_1_a1b2c3d4e5f6789012345678901234ef.csv`
  - platform_id: 1 (Facebook), 2 (Instagram), 3 (Twitter)
  - video_id: 1-3
  - session_id: 1-2

### Sequence Mapping
```
Facebook (f):  [0, 3, 6, 9, 12, 15]
Instagram (i): [1, 4, 7, 10, 13, 16]  
Twitter (t):   [2, 5, 8, 11, 14, 17]
```

### Metadata Files
- `{user_id}_consent.json` - Consent form data
- `{user_id}_demographics.json` - User demographics
- `{user_id}_start_time.json` - Session start time
- `{user_id}_completion.json` - Session completion status (optional)

## Data Formats

### Keystroke CSV Format
```csv
P,h,1000000000
P,e,1100000000
R,h,1150000000
R,e,1250000000
```
- Column 1: Event type (P=Press, R=Release)
- Column 2: Key pressed
- Column 3: Timestamp in nanoseconds

### Keypair Features
```
user_id,platform_id,video_id,session_id,key1,key2,HL,IL,PL,RL,valid,outlier
```
- HL: Hold Latency (key1_release - key1_press)
- IL: Inter-key Latency (key2_press - key1_release)
- PL: Press Latency (key2_press - key1_press)
- RL: Release Latency (key2_release - key1_release)

### Feature Aggregations

#### User-Platform Level
Statistical features (mean, std, median, etc.) aggregated across all videos/sessions for each platform.

#### Session Level
Features aggregated per session (2 sessions per platform).

#### Video Level
Features aggregated per video (3 videos per session).

## Validation Rules

### Complete User Requirements
1. All 18 keystroke files (3 platforms × 3 videos × 2 sessions)
2. Required metadata: consent.json, demographics.json, start_time.json
3. Valid 32-character hex user IDs
4. Proper file naming conventions

### Data Quality Checks
- No negative hold latencies (HL > 0)
- Reasonable timing bounds (30ms < HL < 2s)
- Valid key sequences (press before release)
- No orphan events (unmatched press/release)