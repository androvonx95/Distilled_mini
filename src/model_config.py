from transformers import GPT2Config

def get_config():
    return GPT2Config(
        vocab_size=50257,
        n_positions=2048,
        n_ctx=2048,
        n_embd=512,
        n_layer=6,
        n_head=8,
        pad_token_id=50256  # same as EOS since GPT2 doesn't actually have any padding token
    )


# with these parameters its said to be around 50-60M parameters 
# to get to 100M we need more layers and larger embedding vector size