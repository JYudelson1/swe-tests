from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

conv = [
    {"role": "user", "content": "Hello, how are you?"}, 
    {"role": "assistant", "content": "<think> What should I say? </think>\n\nI'm good, thank you!"}, 
    {"role": "tool", "content": "<return code>0</return_code>\n"}, 
    {"role": "tool", "content": "<return code>1</return_code>\n"}, 
    {"role": "assistant", "content": ""}
    ]

print(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))