from llama_cpp import Llama

model_path = "/home/nike/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
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
