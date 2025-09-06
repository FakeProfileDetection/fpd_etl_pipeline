#!/usr/bin/env python3
"""Test gpt-oss JSON generation capability"""

import json

import requests


def test_json_generation():
    url = "http://UbuntuSungoddess:1234/v1/chat/completions"

    test_text = "I watched Coach Carter and it was inspiring."

    prompt = f"""Rate this text from 0-100 for mentioning these topics:
Text: "{test_text}"

Return ONLY this JSON (no other text):
{{"Coach Carter": <score>, "Oscars Slap": <score>, "Trump-Ukraine Meeting": <score>}}"""

    models = ["gpt-oss-20b", "gemma-3-12b-it@q8_0"]

    for model in models:
        print(f"\n=== Testing {model} ===")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 100,  # Enough tokens for JSON response
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            reasoning = result["choices"][0]["message"].get("reasoning", "")

            print(f"Content: {content}")
            if reasoning:
                print(f"Reasoning: {reasoning}")

            # Try to parse as JSON
            try:
                # Handle markdown wrapped JSON
                import re

                json_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                    print("(Extracted from markdown)")
                else:
                    json_str = content

                parsed = json.loads(json_str)
                print(f"✓ Valid JSON: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")


if __name__ == "__main__":
    test_json_generation()
