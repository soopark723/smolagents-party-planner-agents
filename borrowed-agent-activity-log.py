# Dependencies:
#   pip install langfuse
#   pip install openinference-instrumentation-smolagents
#   pip install "smolagents[toolkit]"   # from_hub loads Alfred's own bundled tools. Earlier
#                                        # versions of this exact task used DuckDuckGoSearchTool-style
#                                        # web search (needs ddgs, covered by the toolkit extra), but
#                                        # trust_remote_code=True means the repo could pull in other
#                                        # runtime deps you can't know about until you run it — install
#                                        # whatever the next ModuleNotFoundError names, same as before.
#
# (run with your project's interpreter, e.g.:
#   C:\Python314\python.exe -m pip install langfuse openinference-instrumentation-smolagents "smolagents[toolkit]")
#
# Also requires these environment variables (e.g. in your .env):
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST   — for get_client()/auth_check()
#   HF_TOKEN (or a prior huggingface_hub login()) — for the InferenceClientModel calls inside Alfred's agent
#
# NOTE: this file must NOT be named langfuse.py — that shadows the real
# langfuse package and breaks the very first import. Keep it as something
# like langfuse_trace.py.

from langfuse import get_client
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from smolagents import CodeAgent

# 1. Verify Langfuse connection
langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# 2. Instrument smolagents so every step gets traced to Langfuse
SmolagentsInstrumentor().instrument()

# 3. Load Alfred, a pre-built agent shared on the Hub, and run it
alfred_agent = CodeAgent.from_hub("sergiopaniego/AlfredAgent", trust_remote_code=True)
result = alfred_agent.run(
    "Give me the best playlist for a party at Wayne's mansion. "
    "The party idea is a 'villain masquerade' theme"
)
print(result)