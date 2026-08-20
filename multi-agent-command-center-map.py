# Dependencies:
#   pip install "smolagents[toolkit,openai]"  # CodeAgent, GoogleSearchTool, VisitWebpageTool,
#                                              # and OpenAIServerModel (needs the openai client lib)
#   pip install python-dotenv                 # for load_dotenv()
#   pip install pillow                        # for PIL.Image
#   pip install plotly shapely geopandas      # used via additional_authorized_imports
#   pip install kaleido                       # plotly's fig.write_image() needs this to save saved_map.png
#   pip install pandas numpy                  # also used via additional_authorized_imports
#                                              # (likely already installed from earlier fixes)
#
# (run with your project's interpreter, e.g.:
#   C:\Python314\python.exe -m pip install "smolagents[toolkit,openai]" python-dotenv pillow plotly shapely geopandas kaleido pandas numpy)
#
# Also requires these environment variables (e.g. in your .env):
#   SERPER_API_KEY   — for GoogleSearchTool(provider="serper") in web_agent
#   OPENAI_API_KEY   — for OpenAIServerModel("gpt-4o", ...) in check_reasoning_and_plot
#   HF_TOKEN         — for InferenceClientModel calls (should already be set from earlier login())

from dotenv import load_dotenv
load_dotenv()  # reads .env in the current directory and sets the env vars

import os

from PIL import Image
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    InferenceClientModel,
    OpenAIServerModel,
    VisitWebpageTool,
    tool,
)
from smolagents.utils import encode_image_base64, make_image_url


# Reused from batmobile.py — consider moving this into a shared tools.py
# that both scripts import from, instead of pasting it twice.
@tool
def calculate_cargo_travel_time(
    origin_coords: tuple[float, float],
    destination_coords: tuple[float, float],
    cruising_speed_kmh: float = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours
    """
    import math

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    EARTH_RADIUS_KM = 6371.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    actual_distance = distance * 1.1
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0
    return round(flight_time, 2)


model = InferenceClientModel(
    "Qwen/Qwen2.5-Coder-32B-Instruct", provider="nscale", max_tokens=8096
)

web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)


def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True

# openai/gpt-oss-120b, deepseek-ai/DeepSeek-R1
manager_agent = CodeAgent(
    model=InferenceClientModel(
        "openai/gpt-oss-120b", provider="novita", max_tokens=8096, timeout=300
    ),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)

manager_agent.visualize()

manager_agent.run("""
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
""")