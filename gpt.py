from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "syvai/llama3-da-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define input prompt
prompt = "Hvem er Frank Hvam?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate text with proper parameters
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prompt)
print(generated_text)
