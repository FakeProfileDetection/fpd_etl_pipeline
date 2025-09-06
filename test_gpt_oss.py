#!/usr/bin/env python3
"""Test different model ID formats for gpt-oss"""

import requests


def test_model_ids():
    url = "http://UbuntuSungoddess:1234/v1/chat/completions"

    # Try different model ID formats
    model_ids = ["gpt-oss-20b", "openai/gpt-oss-20b", "gpt-oss:20b"]

    for model_id in model_ids:
        print(f"\n=== Testing model ID: {model_id} ===")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "What is 2+2? Reply with just the number."}
            ],
            "temperature": 0.1,
            "max_tokens": 50,
        }

        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            msg = result["choices"][0]["message"]

            print(f"Role: {msg.get('role', 'N/A')}")
            print(f"Content: '{msg.get('content', '')}'")
            print(f"Reasoning: '{msg.get('reasoning', '')}'")

            # Show any other fields
            other_fields = {
                k: v
                for k, v in msg.items()
                if k not in ["role", "content", "reasoning", "tool_calls"]
            }
            if other_fields:
                print(f"Other fields: {other_fields}")

        else:
            print(f"Error: {response.text}")


if __name__ == "__main__":
    test_model_ids()
