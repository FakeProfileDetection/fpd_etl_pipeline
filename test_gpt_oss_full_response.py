#!/usr/bin/env python3
"""Investigate full response structure from gpt-oss model"""

import json

import requests


def test_full_response():
    url = "http://UbuntuSungoddess:1234/v1/chat/completions"

    test_text = "I watched Coach Carter and it was inspiring."

    prompt = f"""Rate this text from 0-100 for mentioning these topics:
Text: "{test_text}"

Return ONLY this JSON (no other text):
{{"Coach Carter": <score>, "Oscars Slap": <score>, "Trump-Ukraine Meeting": <score>}}"""

    print("=== Testing gpt-oss-20b with increased max_tokens ===")

    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "Return only valid JSON, no other text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 500,  # Much larger token limit
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()

        print("\n=== FULL RESPONSE STRUCTURE ===")
        print(json.dumps(result, indent=2))

        print("\n=== ANALYZING MESSAGE CONTENT ===")
        msg = result["choices"][0]["message"]

        # Check all fields in the message
        for key, value in msg.items():
            if key == "tool_calls":
                continue  # Skip tool_calls array
            print(f"\n{key}: {repr(value)[:200]}")

            # If it's a string that might contain JSON, try to parse it
            if isinstance(value, str) and value.strip():
                try:
                    parsed = json.loads(value)
                    print(f"  ^ This {key} field contains valid JSON!")
                    print(f"    Parsed: {json.dumps(parsed, indent=4)}")
                except:
                    # Try to find JSON within the string
                    import re

                    json_patterns = [
                        r"\{[^{}]*\}",  # Simple JSON object
                        r"\{.*?\}",  # Any JSON object
                    ]
                    for pattern in json_patterns:
                        matches = re.findall(pattern, value, re.DOTALL)
                        for match in matches:
                            try:
                                parsed = json.loads(match)
                                print(f"  ^ Found JSON within {key}: {match}")
                                break
                            except:
                                continue

        # Check finish_reason
        print(f"\nfinish_reason: {result['choices'][0].get('finish_reason', 'N/A')}")

    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    test_full_response()
