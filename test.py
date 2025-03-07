from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

conv = [
    {"role": "system", "content": "You are a helpful assistant."}, 
    {"role": "user", "content": "Hello, how are you?"}, 
    {"role": "assistant", "content": "<think> What should I say? </think>\n\nI'm good, thank you!"}, 
    {"role": "tool", "content": "<return code>0</return_code>\n"}, 
    {"role": "tool", "content": "<return code>1</return_code>\n"}, 
    ]

print(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))