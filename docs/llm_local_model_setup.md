# Using Local LLM Models with LM Studio

This document explains how to use local LLM models (via LM Studio) instead of OpenAI API for the LLM check stage.

## Overview

The LLM check stage now supports both:
1. **OpenAI API** (default) - Uses OpenAI's cloud models
2. **Local LM Studio** - Uses models running on your local/network machine

## Quick Start

### 1. Start LM Studio with your model

Ensure LM Studio is running with your model loaded:
```bash
# Check if model is loaded
curl http://UbuntuSungoddess:1234/v1/models
```

### 2. Configure for Local Model

Edit `config/.env.local` and uncomment these lines:
```bash
# Use local LM Studio model
LLM_CHECK_USE_LOCAL=true
LLM_CHECK_BASE_URL=http://UbuntuSungoddess:1234/v1
LLM_CHECK_MODEL=openai/gpt-oss-20b
LLM_CHECK_MAX_CONCURRENT=3  # Reduce for local GPU
```

### 3. Run the Pipeline

```bash
python scripts/pipeline/run_pipeline.py --stages llm_check --local-only --with-llm-check
```

## Configuration Options

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_CHECK_USE_LOCAL` | Enable local model mode | `true` or `false` |
| `LLM_CHECK_BASE_URL` | LM Studio API endpoint | `http://localhost:1234/v1` |
| `LLM_CHECK_MODEL` | Model ID to use | `openai/gpt-oss-20b` |
| `LLM_CHECK_MAX_CONCURRENT` | Concurrent requests | `3` (for local GPU) |

## Switching Between Modes

### Use Local Model
```bash
# In config/.env.local
LLM_CHECK_USE_LOCAL=true
LLM_CHECK_BASE_URL=http://UbuntuSungoddess:1234/v1
LLM_CHECK_MODEL=openai/gpt-oss-20b
```

### Use OpenAI (Default)
```bash
# In config/.env.local
LLM_CHECK_USE_LOCAL=false
# Or just comment out the local settings
```

## Testing Both Modes

Run the test script to verify both modes work:
```bash
python test_llm_modes.py
```

## Supported Models

### Currently Tested
- `openai/gpt-oss-20b` - 20B parameter model

### Future Models
- `openai/gpt-oss-120b` - Coming soon (larger, more capable)
- Any OpenAI-compatible model in LM Studio

## Advantages of Local Models

1. **Privacy** - Data never leaves your network
2. **Cost** - No API usage fees
3. **Control** - Full control over model behavior
4. **Offline** - Works without internet connection

## Disadvantages

1. **Performance** - May be slower than cloud API
2. **Quality** - Results may vary from OpenAI models
3. **Resources** - Requires powerful GPU
4. **Concurrency** - Limited by GPU memory

## Troubleshooting

### Model Not Found
```
Error: Model 'openai/gpt-oss-20b' not found
```
**Solution**: Ensure the model is loaded in LM Studio

### Connection Refused
```
Error: Connection refused to http://UbuntuSungoddess:1234
```
**Solution**:
- Check LM Studio is running
- Verify the hostname/IP is correct
- Check firewall settings

### Out of Memory
```
Error: CUDA out of memory
```
**Solution**:
- Reduce `LLM_CHECK_MAX_CONCURRENT` (try 1 or 2)
- Use a smaller model
- Close other GPU applications

### Poor Results
If the local model gives poor scores:
- Try adjusting the prompt (may need model-specific tuning)
- Consider using a larger model (gpt-oss-120b when available)
- Fall back to OpenAI for production use

## Implementation Details

The implementation:
1. Checks `LLM_CHECK_USE_LOCAL` environment variable
2. If true, uses `LLM_CHECK_BASE_URL` as the API endpoint
3. Sends requests using OpenAI-compatible format
4. Handles responses the same as OpenAI

The code is backward compatible - colleagues without access to your local model can still use OpenAI API normally.

## Best Practices

1. **Development**: Use local model for testing and development
2. **Production**: Use OpenAI for final validation
3. **Hybrid**: Use local for initial screening, OpenAI for edge cases
4. **Model Selection**: Start with smaller models, upgrade as needed

## Future Enhancements

- Support for multiple local models
- Automatic model selection based on task
- Model performance comparison tools
- Prompt optimization for specific models
