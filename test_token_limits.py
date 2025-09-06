#!/usr/bin/env python3
"""Test how token limits affect gpt-oss output"""

import json

import requests


def test_token_limits():
    url = "http://UbuntuSungoddess:1234/v1/chat/completions"

    test_text = "I watched Coach Carter and it was inspiring."

    prompt = f"""Rate this text from 0-100 for mentioning these topics:
Text: "{test_text}"

Return ONLY this JSON (no other text):
{{"Coach Carter": <score>, "Oscars Slap": <score>, "Trump-Ukraine Meeting": <score>}}"""

    token_limits = [50, 100, 200, 500]

    for max_tokens in token_limits:
        print(f"\n=== Testing with max_tokens={max_tokens} ===")

        payload = {
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "Return only valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            msg = result["choices"][0]["message"]

            content = msg.get("content", "")
            reasoning = msg.get("reasoning", "")
            finish_reason = result["choices"][0].get("finish_reason", "N/A")

            print(f"Finish reason: {finish_reason}")
            print(f"Content length: {len(content)} chars")
            print(f"Content: {content[:100]!r}")

            if content.strip():
                try:
                    parsed = json.loads(content)
                    print("✓ Valid JSON in content field")
                except:
                    print("✗ Content is not valid JSON")
            else:
                print("✗ Content is empty")

            if reasoning:
                print(f"Reasoning (first 100 chars): {reasoning[:100]!r}")


if __name__ == "__main__":
    test_token_limits()
