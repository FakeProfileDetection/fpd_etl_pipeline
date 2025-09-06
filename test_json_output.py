#!/usr/bin/env python3
"""Test JSON output from local LLM model"""

import json

import requests


def test_json_output():
    url = "http://UbuntuSungoddess:1234/v1/chat/completions"

    # Try both models
    models = ["openai/gpt-oss-20b", "gemma-3-12b-it@q8_0"]

    for model in models:
        print(f"\n=== Testing {model} ===")
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is 2+2? Reply with just the number."}
            ],
            "temperature": 0.1,
            "max_tokens": 10,
        }

        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Full response: {json.dumps(result, indent=2)}")

            if "choices" in result and len(result["choices"]) > 0:
                msg = result["choices"][0]["message"]
                content = msg.get("content", "")
                reasoning = msg.get("reasoning", "")

                print(f"Content: '{content}'")
                print(f"Reasoning: '{reasoning}'")

                # Check if response is in content or reasoning field
                actual_response = content if content else reasoning
                if actual_response:
                    print(f"Actual response: '{actual_response}'")
                else:
                    print("Both content and reasoning are empty")
        else:
            print(f"Error: {response.text}")


if __name__ == "__main__":
    test_json_output()
