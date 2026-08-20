# Dependencies:
#   pip install pillow requests "smolagents[openai]"
#
# (run with your project's interpreter, e.g.:
#   C:\Python314\python.exe -m pip install pillow requests "smolagents[openai]")
#
# Also requires OPENAI_API_KEY set as an environment variable — OpenAIServerModel
# talks to OpenAI directly, not through Hugging Face.

from io import BytesIO

import requests
from PIL import Image
from smolagents import CodeAgent, OpenAIServerModel

IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg",
]

# Wikimedia blocks the default requests User-Agent, so we spoof a browser one.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}


def download_images(urls: list[str]) -> list[Image.Image]:
    images = []
    for url in urls:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        images.append(Image.open(BytesIO(response.content)).convert("RGB"))
    return images


def main():
    images = download_images(IMAGE_URLS)

    model = OpenAIServerModel(model_id="gpt-4o")
    agent = CodeAgent(tools=[], model=model, max_steps=20, verbosity_level=2)

    response = agent.run(
        """
        Describe the costume and makeup that the comic character in these photos is wearing and return the description.
        Tell me if the guest is The Joker or Wonder Woman.
        """,
        images=images,
    )
    print(response)


if __name__ == "__main__":
    main()