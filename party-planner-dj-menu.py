# Dependencies:
#   pip install huggingface_hub
#   pip install "smolagents[toolkit]"
#
# (run with your project's interpreter, e.g.:
#   C:\Python314\python.exe -m pip install huggingface_hub "smolagents[toolkit]")

from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool

MODEL_ID = "openai/gpt-oss-120b"

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.

    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    menus = {
        "casual": "Pizza, snacks, and drinks.",
        "formal": "3-course dinner with wine and dessert.",
        "superhero": "Buffet with high-energy and healthy food.",
    }
    return menus.get(occasion, "Custom menu for the butler.")


def main():
    login()  # no-op if you're already logged in

    model = InferenceClientModel(model_id=MODEL_ID)

    music_agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
    try:
        music = music_agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
        print("🎵 Music:", music)
    except Exception as e:
        print("🎵 Music agent failed:", e)

    menu_agent = CodeAgent(tools=[suggest_menu], model=model)
    try:
        menu = menu_agent.run("Prepare a formal menu for the party.")
        print("🍽️ Menu:", menu)
    except Exception as e:
        print("🍽️ Menu agent failed:", e)


if __name__ == "__main__":
    main()