import argparse
import json
import os
from time import sleep
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError


def get_txt_files(folder_path: str) -> List[str]:
    return [f for f in os.listdir(folder_path) if f.endswith(".txt")]


def read_file_content(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return file.read()


def build_prompt(text: str) -> str:
    return f"""
You are a helpful AI tasked with determining relevance.

Given the following text:

{text}

Rate on a scale from 0 to 100 how relevant this text is to each of the following topics:

1. Coach Carter (2005 movie)
2. The 2022 Oscars slap (Will Smith and Chris Rock)
3. The Trump–Zelenskyy White House meeting

Return your answer as JSON in the format:
{{
    "Coach Carter": <percentage>,
    "Oscars Slap": <percentage>,
    "Trump-Ukraine Meeting": <percentage>
}}
ONLY return the JSON object, nothing else.
Do not include any other text or explanation.

Determine the relevance of the text to each topic and return a percentage score for
each topic. The comments must not only discuss the video but comment on the video.
For example, ONLY stating the name of the video is NOT relevant. The must DISCUSS
the video in some way.  The comments must be relevant to the video and not just a
general comment.

"""


def analyze_text(text: str, model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"):
    # Initialize Bedrock client
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    max_tokens = 4096
    temperature: float = 0.5
    top_p: float = 0.9
    # Construct the request payload (using 'body' instead of 'prompt')
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": "You are a helpful AI tasked with determining relevance of text to specific movies and events.",
        "top_p": top_p,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(text)},
                ],
            }
        ],
    }
    try:
        response = client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        response_body: dict = json.loads(response["body"].read())
        completion: str = response_body["content"][0]["text"]
        print(f"Model response: {completion}")
        print("___________________________________________")

        usage_info: Optional[dict] = response_body.get("usage")
        input_tokens: Optional[int] = (
            usage_info.get("input_tokens") if usage_info else None
        )
        output_tokens: Optional[int] = (
            usage_info.get("output_tokens") if usage_info else None
        )

        if usage_info:
            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
        else:
            print("No usage information available in the response.")

        return (completion, input_tokens, output_tokens)

    except ClientError as e:
        raise ValueError(
            f"Error calling Bedrock model: {e.response['Error']['Message']}"
        )


def main(folder_path: str, args):
    model = args.model
    output_file = args.output
    json_data = []
    files = get_txt_files(folder_path)
    for filename in files:
        try:
            path = os.path.join(folder_path, filename)
            text = read_file_content(path)
            print(f"\nAnalyzing {filename}...")
            result_str, _, _ = analyze_text(text, model=model)
            result = json.loads(result_str)
            result["filename"] = filename
            result["text"] = str(text)
            json_data.append(result)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("Could not determine relevance.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print("Skipping this file.")
        sleep(1)

    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check text relevance to specific events."
    )
    parser.add_argument("folder", help="Folder containing .txt files")
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="Bedrock model to use (default: us.anthropic.claude-sonnet-4-20250514-v1:0)",
    )
    parser.add_argument(
        "-o", "--output", help="Output file for results", default="results.json"
    )
    args = parser.parse_args()

    main(args.folder, args)

# def main(folder_path: str, model: str = "anthropic/claude-v1"):
#     files = get_txt_files(folder_path)
#     for filename in files:
#         sleep(30)
#         path = os.path.join(folder_path, filename)
#         text = read_file_content(path)
#         print(f"\nAnalyzing {filename}...")
#         result = analyze_text(text, model=model)
#         sleep(30)  # Sleep to avoid rate limiting
#         if result:
#             print(json.dumps(result, indent=2))
#         else:
#             print("Could not determine relevance.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Check text relevance to specific events."
#     )
#     parser.add_argument("folder", help="Folder containing .txt files")
#     parser.add_argument(
#         "--model",
#         default="us.anthropic.claude-sonnet-4-20250514-v1:0",
#         help="Bedrock model to use (default: us.anthropic.claude-sonnet-4-20250514-v1:0)",
#     )
#     args = parser.parse_args()

#     main(args.folder, model=args.model)
