from transformers import AutoTokenizer, AutoModelForCausalLM , LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")


model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

breakpoint()

a=1
b=2