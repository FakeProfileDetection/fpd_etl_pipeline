#!/usr/bin/env python3
"""
Test script to verify both OpenAI and LM Studio modes work
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# Test both configurations
async def test_llm_modes():
    """Test both OpenAI and local LM Studio modes"""

    print("=" * 80)
    print("LLM MODE TESTING")
    print("=" * 80)

    # Test sample text
    test_text = """
    I watched the Coach Carter video and it was really inspiring.
    The student's speech about our deepest fear was powerful.
    It made me think about my own potential.
    """

    # 1. Test OpenAI mode (if API key exists)
    print("\n1. Testing OpenAI mode...")
    print("-" * 40)

    # Clear local mode settings
    os.environ.pop("LLM_CHECK_USE_LOCAL", None)
    os.environ.pop("LLM_CHECK_BASE_URL", None)
    os.environ.pop("LLM_CHECK_MODEL", None)

    from scripts.pipeline.extract_llm_scores import OpenAIProcessor

    # Check if OpenAI API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            processor = OpenAIProcessor(
                api_key=api_key, model="gpt-4o-mini", max_concurrent=1
            )

            # Process as a batch with one item
            results = await processor.process_batch(
                [
                    (
                        test_text,
                        {
                            "user_id": "test_user",
                            "device_type": "desktop",
                            "platform_id": 1,
                            "video_id": 1,
                            "session_id": 1,
                            "filename": "test.txt",
                        },
                    )
                ]
            )
            result = results[0] if results else {}

            print("✓ OpenAI mode works!")
            print(f"  Coach Carter Score: {result.get('Coach Carter', 0)}%")
            print(f"  Oscars Slap Score: {result.get('Oscars Slap', 0)}%")
            print(f"  Trump-Ukraine Score: {result.get('Trump-Ukraine Meeting', 0)}%")

        except Exception as e:
            print(f"✗ OpenAI mode failed: {e}")
    else:
        print("⚠ OpenAI API key not found, skipping OpenAI test")

    # 2. Test LM Studio mode
    print("\n2. Testing LM Studio mode...")
    print("-" * 40)

    # Set local mode configuration
    os.environ["LLM_CHECK_USE_LOCAL"] = "true"
    os.environ["LLM_CHECK_BASE_URL"] = "http://UbuntuSungoddess:1234/v1"
    os.environ["LLM_CHECK_MODEL"] = "openai/gpt-oss-20b"

    # Reload the module to pick up new env vars
    import importlib

    import scripts.pipeline.extract_llm_scores

    importlib.reload(scripts.pipeline.extract_llm_scores)
    from scripts.pipeline.extract_llm_scores import OpenAIProcessor

    try:
        processor = OpenAIProcessor(
            api_key="dummy-key-for-local",
            model="gpt-oss-20b",  # Use model ID without openai/ prefix
            max_concurrent=1,
            base_url="http://UbuntuSungoddess:1234/v1",
            is_local=True,
        )

        # Process as a batch with one item
        results = await processor.process_batch(
            [
                (
                    test_text,
                    {
                        "user_id": "test_user",
                        "device_type": "desktop",
                        "platform_id": 1,
                        "video_id": 1,
                        "session_id": 1,
                        "filename": "test.txt",
                    },
                )
            ]
        )
        result = results[0] if results else {}

        print("✓ LM Studio mode works!")
        print(f"  Coach Carter Score: {result.get('Coach Carter', 0)}%")
        print(f"  Oscars Slap Score: {result.get('Oscars Slap', 0)}%")
        print(f"  Trump-Ukraine Score: {result.get('Trump-Ukraine Meeting', 0)}%")

    except Exception as e:
        print(f"✗ LM Studio mode failed: {e}")
        print("  Make sure LM Studio is running at http://UbuntuSungoddess:1234")
        print("  And that the model 'openai/gpt-oss-20b' is loaded")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("\nTo use LM Studio in the pipeline, add these settings to .env:")
    print("  LLM_CHECK_USE_LOCAL=true")
    print("  LLM_CHECK_BASE_URL=http://UbuntuSungoddess:1234/v1")
    print("  LLM_CHECK_MODEL=openai/gpt-oss-20b")
    print("  LLM_CHECK_MAX_CONCURRENT=3")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_llm_modes())
