"""Quick smoke test of llama-cpp-python with a GGUF model.

Set JOURNAL_CLUB_GGUF_MODEL_PATH before running.
"""
import os

from llama_cpp import Llama

model_path = os.getenv("JOURNAL_CLUB_GGUF_MODEL_PATH")
if not model_path:
    raise SystemExit("Set JOURNAL_CLUB_GGUF_MODEL_PATH to a GGUF model file first.")

print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # use GPU if available
    n_ctx=4096,
)
print("Success!")

# Example usage
output = llm("Hello, world!", max_tokens=100)
print(output["choices"][0]["text"])
