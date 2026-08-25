# smolagents-party-planner-agents

A collection of [smolagents](https://github.com/huggingface/smolagents) examples built while working through Hugging Face's [Agents Course](https://huggingface.co/learn/agents-course). The running theme: Alfred, the butler of Wayne Manor, uses a growing toolkit of AI agents to plan a superhero-themed party – searching the web, generating images, mapping logistics, checking guests' identities, and more.

## Scripts

| File | What it does |
|---|---|
| `party-planner-dj-menu.py` | Two simple agents: one searches the web for music recommendations, the other picks a menu from a small lookup table based on the party's vibe. |
| `world-tour-logistics.py` | Searches the web for real Batman filming locations and supercar factories, then uses a hand-written flight-time calculator to figure out cargo plane transfer times to Gotham. Runs a quick pass, then a slower, more thorough one. |
| `poster-generator.py` | Expands a short prompt into a richer image description, then generates a party poster using the FLUX.1-schnell model via a Hugging Face Space. |
| `multi-agent-command-center-map.py` | A "manager" agent delegates web research to a "web" agent, plots every location on a world map colored by travel time, and has a second AI (GPT-4o) review the finished map before accepting it. |
| `local-party-idea-search.py` | Retrieves relevant party-planning ideas from a small local knowledge base using BM25 keyword search (via LangChain), no internet required. |
| `borrowed-agent-activity-log.py` | Loads a pre-built agent from the Hugging Face Hub and wires up Langfuse to trace every step it takes. |
| `verify-image.py` | Downloads reference photos and asks GPT-4o to describe the costume/makeup shown, to help verify a guest's claimed identity. |

> Filenames above match what was built during development – rename to match your actual files if they differ.

## Setup

Each script lists its own dependencies in a comment block at the top of the file – install those before running a given script. In general, across the whole repo you'll need:

```powershell
pip install smolagents huggingface_hub python-dotenv
```

plus whichever extras a specific script calls for (`smolagents[toolkit]`, `smolagents[openai]`, `pandas`, `plotly`, `geopandas`, `shapely`, `kaleido`, `langchain`, `langchain_community`, `rank_bm25`, `helium`, `selenium`, `langfuse`, `openinference-instrumentation-smolagents`, etc.).

### Environment variables

Create a `.env` file in the project root (already gitignored) with whichever of these the scripts you're running need:

```dotenv
HF_TOKEN=your_huggingface_token
SERPER_API_KEY=your_serper_api_key
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host
```

### Other prerequisites

- `wonder_woman_browser_agent.py` needs **Google Chrome** installed on your machine.
- Hugging Face Inference Providers change which models they serve over time – if you hit a `model_not_supported` error, check the model's page on huggingface.co for its current provider(s) and update the `provider=` argument accordingly.

## Running a script

```powershell
python path/to/script.py
```

## Acknowledgments

Built while following Hugging Face's [Agents Course](https://huggingface.co/learn/agents-course) (smolagents unit) – several scripts adapt examples and the Alfred/Wayne Manor scenario from that material.
