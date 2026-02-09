'''
This script serves as a utility to upload to Label Studio
JSON files that exceed the maximum size of the Web App GUI. 
'''

import json
import requests
import os

LS_URL = "http://localhost:8080"
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
PROJECT_ID = 19
TASKS_FILE = "/mnt/d/Dataset/LabelStudio/RailData_valid_ls/task.json"
CHUNK_SIZE = 800

headers = {
    "Authorization": f"Token {API_KEY}"
}

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def import_tasks():
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    chunked_tasks = chunked(tasks, CHUNK_SIZE)

    for idx, batch in enumerate(chunked_tasks, start=1):
        print(f"Uploading chunk {idx} with {len(batch)} tasks...")

        res = requests.post(
            f"{LS_URL}/api/projects/{PROJECT_ID}/import",
            headers=headers,
            json=batch
        )

        if res.status_code == 201:
            print(f"Chunk {idx} uploaded successfully")
        elif res.status_code == 401:
            print(f"Unauthorized. API KEY: {API_KEY}")
        else:
            print(f"Error uploading chunk {idx}: {res.status_code}, {res.text}")

if __name__ == "__main__":
    import_tasks()
