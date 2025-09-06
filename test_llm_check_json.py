#!/usr/bin/env python3
"""Test JSON output for LLM check task with Gemma model"""

import json

import requests


def test_llm_check_json():
    url = "http://UbuntuSungoddess:1234/v1/chat/completions"

    test_text = """
    I watched the Coach Carter movie and it was really inspiring.
    The coach's dedication to his students was amazing.
    """

    prompt = f"""Evaluate this text and provide scores from 0-100 for how much it mentions each topic:

Text: {test_text}

Return ONLY a JSON object with these exact keys:
{{"Coach Carter": <score>, "Oscars Slap": <score>, "Trump-Ukraine Meeting": <score>}}

Example: {{"Coach Carter": 90, "Oscars Slap": 0, "Trump-Ukraine Meeting": 0}}"""

    payload = {
        "model": "gemma-3-12b-it@q8_0",
        "messages": [
            {
                "role": "system",
                "content": "You are a precise evaluator. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 100,
    }

    print("Testing Gemma model for LLM check JSON generation...")
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\nRaw response:\n{content}")

        # Try to extract JSON from markdown code blocks
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"\nExtracted JSON from markdown: {json_str}")
        else:
            json_str = content

        try:
            parsed = json.loads(json_str)
            print("\n✓ Successfully parsed JSON:")
            print(json.dumps(parsed, indent=2))

            # Validate structure
            required_keys = {"Coach Carter", "Oscars Slap", "Trump-Ukraine Meeting"}
            if set(parsed.keys()) == required_keys:
                print("\n✓ All required keys present!")
                print(f"  Coach Carter: {parsed['Coach Carter']}%")
                print(f"  Oscars Slap: {parsed['Oscars Slap']}%")
                print(f"  Trump-Ukraine: {parsed['Trump-Ukraine Meeting']}%")
            else:
                print(f"\n✗ Missing keys. Got: {parsed.keys()}")

        except json.JSONDecodeError as e:
            print(f"\n✗ JSON parsing failed: {e}")
            print("This model may not be suitable for the LLM check task")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    test_llm_check_json()
