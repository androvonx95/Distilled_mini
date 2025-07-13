from transformers import GPT2Config

def get_config():
    return GPT2Config(
        vocab_size=50257,
        n_positions=2048,
        n_ctx=2048,
        n_embd=1024,  # Increased from 768
        n_layer=16,   # Increased from 12
        n_head=16,    # Increased from 12
        pad_token_id=50256,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True
    )


# This configuration results in approximately 112M parameters
# To reach approximately 150M parameters:
# - Increase n_embd to 1024
# - Or increase n_head to 16
# - Or add more layers (though more than 12 is not recommended)