from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("Hello, This is an example of tokenization!")
print("Tokens:", tokens)
print("Tokens length:", len(tokens))