# Dependencies:
#   pip install "smolagents[toolkit]"   # CodeAgent, tool decorator, GoogleSearchTool,
#                                        # and VisitWebpageTool (which needs markdownify,
#                                        # bundled in the toolkit extra)
#   pip install pandas                  # used via additional_authorized_imports
#   pip install python-dotenv           # for load_dotenv()
#
# (run with your project's interpreter, e.g.:
#   C:\Python314\python.exe -m pip install "smolagents[toolkit]" pandas python-dotenv)
#
# Also requires SERPER_API_KEY set in your .env (for GoogleSearchTool("serper"))

from dotenv import load_dotenv
load_dotenv()  # reads .env in the current directory and sets the env vars

import math
from typing import Optional, Tuple

from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel, VisitWebpageTool, tool


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,  # Average speed for cargo planes
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

    actual_distance = distance * 1.1  # non-direct routes / ATC buffer
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0  # + takeoff/landing
    return round(flight_time, 2)


if __name__ == "__main__":
    print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))

    # "nscale" is the provider that actually serves this model right now.
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="nscale")

    task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time."""

    agent = CodeAgent(
        model=model,
        tools=[GoogleSearchTool("serper"), VisitWebpageTool(), calculate_cargo_travel_time],
        additional_authorized_imports=["pandas"],
        max_steps=20,
    )

    result = agent.run(task)
    print(result)

    agent.planning_interval = 4

    detailed_report = agent.run(f"""
You're an expert analyst. You make comprehensive reports after visiting many websites.
Don't hesitate to search for many queries at once in a for loop.
For each data point that you find, visit the source url to confirm numbers.

{task}
""")

    print(detailed_report)