# LLM Check Integration Documentation

## Overview
The LLM Check stage is an **optional** pipeline component that validates user text responses using OpenAI's API to ensure users properly engaged with video content. This is critical for MTurk payment validation.

## Purpose
- **Validate user engagement**: Ensure users watched and commented on videos appropriately
- **MTurk payment decisions**: Determine which users should be paid based on response quality
- **Quality assurance**: Identify low-quality or fraudulent responses
- **Research data integrity**: Ensure collected data meets research standards

## Requirements

### Required for LLM Check
- OpenAI API key (get from https://platform.openai.com/api-keys)
- Python packages: `openai`, `aiofiles`, `tqdm` (install with `pip install openai aiofiles tqdm`)

### Optional (LLM check can be skipped)
- Team members without API keys can still run the full pipeline
- LLM check results can be shared via reports

## Usage

### Command Line Options

```bash
# New flags added to run_pipeline.py
--with-llm-check        # Enable LLM check stage (default: disabled)
--non-interactive       # Skip prompts, fail if API key missing (for CI/CD)
```

### Running the Pipeline

#### With LLM Check (Interactive)
```bash
# First time - will prompt for API key if not found
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check

# The pipeline will:
# 1. Check for API key in .env, config/.env.local, or environment
# 2. If not found, prompt with options:
#    - Enter API key (saves to .env)
#    - Skip LLM check
#    - Exit to add manually
```

#### With LLM Check (Non-Interactive/CI)
```bash
# Set API key in environment
export OPENAI_API_KEY=sk-...

# Run with non-interactive flag
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check --non-interactive

# If API key missing, stage will be skipped with warning
```

#### Without LLM Check (Default)
```bash
# Pipeline runs normally without LLM validation
python scripts/pipeline/run_pipeline.py --mode full

# No errors, no prompts, pipeline completes normally
```

#### Run Only LLM Check
```bash
# Run LLM check on existing cleaned data
python scripts/pipeline/run_pipeline.py --stages llm_check --local-only

# Useful for:
# - Re-running with different thresholds
# - Processing after initial pipeline run
# - Testing API key setup
```

## API Key Setup

### Method 1: Interactive Setup
Let the pipeline prompt you when running with `--with-llm-check`

### Method 2: Manual Setup
Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### Method 3: Environment Variable
```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

### Method 4: Config File
Add to `config/.env.local`:
```bash
OPENAI_API_KEY=sk-your-api-key-here
LLM_CHECK_MODEL=gpt-4o-mini      # Optional: model selection
LLM_CHECK_MAX_CONCURRENT=5        # Optional: parallel API calls (reduce if rate limited)
LLM_CHECK_THRESHOLD=40            # Optional: pass/fail threshold (0-100)
```

### Rate Limiting Protection
The pipeline includes robust handling for OpenAI rate limits:
- **Automatic retries**: Up to 5 attempts with exponential backoff
- **Smart delays**: Longer waits for rate limit errors (429)
- **Verification**: Checks all files are processed
- **Error handling**: Failed requests default to score 0 with warnings

If you encounter persistent rate limits:
1. Reduce `LLM_CHECK_MAX_CONCURRENT` to 3 or lower
2. Re-run the pipeline - it will retry failed requests
3. Check the logs for "API errors" count

## Output Files

The LLM check stage creates the following outputs in `artifacts/{version_id}/llm_scores/`:

### 1. scores.csv
Main results file with columns:
- `user_id` - User identifier
- `device_type` - desktop/mobile
- `platform_id` - 1=Facebook, 2=Instagram, 3=Twitter
- `video_id` - Video number (1-3)
- `session_id` - Session number (1-2)
- `filename` - Original text filename
- `coach_carter_score` - Relevance score (0-100)
- `oscars_slap_score` - Relevance score (0-100)
- `trump_ukraine_score` - Relevance score (0-100)
- `max_score` - Highest of the 3 scores
- `likely_video` - Which video user likely watched
- `engagement_level` - HIGH/MODERATE/MINIMAL/NONE
- `passes_threshold` - Boolean (max_score >= threshold)
- `text_preview` - First 100 characters

### 2. scores.json
Detailed JSON with:
- All CSV data
- Full text responses
- User-level statistics
- Processing metadata

### 3. reports/summary_report.html
Interactive HTML report with:
- Summary statistics dashboard
- Searchable user table with failed response filter
- Pass/fail rates
- Score distributions
- Click-to-view full responses with modal popups
- Detailed inspection for failed users

### 4. reports/flagged_users.csv
List of users failing validation:
- User IDs
- Response counts
- Average scores
- Useful for MTurk rejection lists

### 5. metadata/
- `processing_stats.json` - API usage, timing, errors
- `skipped.flag` - Created if stage was skipped (no API key)

### 6. broken_data/ (if broken users exist)
Separate analysis for users with incomplete data:
- `scores.csv` - Scoring results for broken users only
- `scores.json` - Detailed results with full text
- `reports/summary_report.html` - Separate report for broken users
- Useful for payment decisions when data is partially lost

## Scoring Logic

The LLM evaluates each text response against 3 video categories:

1. **Coach Carter** - Basketball coach inspirational speech
2. **Oscars Slap** - Will Smith slapping Chris Rock
3. **Trump-Ukraine Meeting** - 2019 diplomatic meeting

### Scoring Guidelines
- **80-100%**: High engagement - specific references, emotions, opinions
- **60-79%**: Moderate engagement - general but relevant discussion
- **40-59%**: Minimal engagement - vague references
- **0-39%**: No/fake engagement - complaints, gibberish, manipulation

### Pass/Fail Logic
- Individual response passes if `max_score >= 40` (configurable)
- User passes overall if ≥14 out of 18 responses pass (>75%)

## Integration with Existing Tools

### Comparison with llm_check.py (Ollama)
- `llm_check.py` - Uses local Ollama models (free, slower)
- `extract_llm_scores.py` - Uses OpenAI API (paid, faster, more accurate)
- Both use similar prompts for consistency

### Compatibility with openai_batched.py
- Core logic adapted from `openai_batched.py`
- Enhanced with pipeline integration
- Maintains backward compatibility for standalone use

## Team Collaboration

### For Teams with Mixed API Access

**Team members WITH API keys:**
```bash
# Run full validation
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check

# Share results via:
# - Upload artifacts to GCS
# - Share HTML reports
# - Commit scores.csv to repo
```

**Team members WITHOUT API keys:**
```bash
# Run pipeline without LLM check
python scripts/pipeline/run_pipeline.py --mode full

# Can still:
# - Process keystroke data
# - Generate features
# - Run EDA
# - Use shared LLM results from teammates
```

## Troubleshooting

### "OpenAI library not installed"
```bash
pip install openai aiofiles tqdm
```

### "API key not found"
- Check `.env` file exists and contains `OPENAI_API_KEY=sk-...`
- Ensure key starts with `sk-`
- Try `echo $OPENAI_API_KEY` to verify environment

### "Rate limit exceeded"
- Reduce concurrent requests: `LLM_CHECK_MAX_CONCURRENT=5`
- Upgrade OpenAI plan for higher limits

### Stage skipped unexpectedly
- Check `artifacts/{version_id}/llm_scores/metadata/skipped.flag`
- Verify `--with-llm-check` flag was used
- Check API key is properly configured

## Enhanced User Inspection Features

The HTML report includes advanced inspection capabilities:

1. **Interactive User Table**
   - Click any user ID to view detailed responses
   - Sort by any column (user ID, responses, pass rate, avg score)
   - Color-coded status indicators

2. **Failed Response Filtering**
   - Search box filters users in real-time
   - "Show Failed Only" checkbox for quick filtering
   - Full text displayed for all failed responses

3. **Detailed Modal View**
   - Click user ID to open modal with all responses
   - Color-coded score badges (red/yellow/green)
   - Full text for failed responses
   - Text preview for passing responses

4. **Broken User Analysis**
   - Separate report for incomplete data
   - Helps with payment decisions when data is lost
   - Same inspection features as complete users

## Best Practices

1. **Cost Management**
   - Use `gpt-4o-mini` model (default) for cost efficiency
   - Process in batches to monitor costs
   - Set up usage limits in OpenAI dashboard

2. **Quality Assurance**
   - Review flagged users manually for edge cases
   - Adjust threshold based on manual validation
   - Compare results with `llm_check.py` for consistency

3. **Pipeline Integration**
   - Run LLM check after data cleaning for best results
   - Can run independently of other feature extraction
   - Results don't affect downstream stages

4. **Data Privacy**
   - Text responses sent to OpenAI API
   - No PII should be in text responses
   - Use `--include-pii=false` (default) for safety

## Example Workflows

### Full Pipeline with Validation
```bash
# Download, clean, validate, and process
python scripts/pipeline/run_pipeline.py \
    --mode full \
    --with-llm-check \
    --upload-artifacts
```

### Validate Existing Data
```bash
# Just run LLM check on previously cleaned data
python scripts/pipeline/run_pipeline.py \
    --stages llm_check \
    --local-only \
    --version-id 2025-08-15_10-00-00_hostname
```

### CI/CD Integration
```bash
# In GitHub Actions or similar
export OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
python scripts/pipeline/run_pipeline.py \
    --mode full \
    --with-llm-check \
    --non-interactive \
    --upload-artifacts
```

## Summary

The LLM Check integration provides:
- **Optional validation** - doesn't break existing workflows
- **Flexible API key handling** - interactive or automated
- **Comprehensive reporting** - HTML, CSV, and JSON outputs
- **MTurk integration** - Clear pass/fail decisions
- **Team-friendly** - Works with or without API access

This ensures high-quality data collection while maintaining pipeline flexibility for all team members.
