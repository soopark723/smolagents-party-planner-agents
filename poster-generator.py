# Dependencies:
#   pip install smolagents
#   pip install "gradio_client<2.0"   # Tool.from_space needs gradio_client; smolagents
#                                      # still calls the old `hf_token` kwarg, so pin below 2.0
#                                      # (or try "pip install -U smolagents" first, in case
#                                      # a newer release has already switched to `token=`)
#
# (run with your project's interpreter, e.g.:
#   C:\Python314\python.exe -m pip install smolagents "gradio_client<2.0")

from smolagents import CodeAgent, InferenceClientModel, Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider="nscale")

agent = CodeAgent(tools=[image_generation_tool], model=model)

result = agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={'user_prompt': 'A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala'}
)

print(result)
if hasattr(result, "save"):
    result.save("generated_image.png")
    print("Saved to generated_image.png")