import base64
import os

import requests
from pyrootutils import setup_root

ROOT = setup_root(".", dotenv=True)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def request_gpt4v(image_path, prompt: str):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        f"{os.getenv('OPENAI_API_BASE')}/chat/completions", headers=headers, json=payload
    )

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(request_gpt4v(ROOT / ".temp" / "bb8ce653.png", "图片里有什么？"))
