from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "/scratch/fbsnpat/bot/journal_club/mistral-7b"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # or "cuda:0" if you have a specific GPU
    torch_dtype="auto"
)
print("Success!")