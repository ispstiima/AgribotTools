"""
Utility to upload JSON task files to Label Studio.

Handles files that exceed the maximum upload size of the Label Studio
Web UI by splitting them into configurable chunks.
"""

import json
import os
import argparse
import requests


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def import_tasks(ls_url, api_key, project_id, tasks_file, chunk_size):
    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    headers = {
        "Authorization": f"Token {api_key}"
    }

    chunked_tasks = chunked(tasks, chunk_size)

    for idx, batch in enumerate(chunked_tasks, start=1):
        print(f"Uploading chunk {idx} with {len(batch)} tasks...")

        res = requests.post(
            f"{ls_url}/api/projects/{project_id}/import",
            headers=headers,
            json=batch
        )

        if res.status_code == 201:
            print(f"Chunk {idx} uploaded successfully")
        elif res.status_code == 401:
            print(f"Unauthorized. Check your API key.")
        else:
            print(f"Error uploading chunk {idx}: {res.status_code}, {res.text}")


def main():
    parser = argparse.ArgumentParser(
        description='Upload JSON task files to Label Studio in chunks'
    )
    parser.add_argument('tasks_file', type=str,
                        help='Path to the JSON tasks file to upload')
    parser.add_argument('--project_id', type=int, required=True,
                        help='Label Studio project ID')
    parser.add_argument('--ls_url', type=str, default='http://localhost:8080',
                        help='Label Studio server URL (default: http://localhost:8080)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Label Studio API key (default: reads LABEL_STUDIO_API_KEY env var)')
    parser.add_argument('--chunk_size', type=int, default=800,
                        help='Number of tasks per upload chunk (default: 800)')
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("LABEL_STUDIO_API_KEY")
    if not api_key:
        parser.error("API key required: use --api_key or set LABEL_STUDIO_API_KEY env var")

    import_tasks(args.ls_url, api_key, args.project_id, args.tasks_file, args.chunk_size)


if __name__ == "__main__":
    main()
