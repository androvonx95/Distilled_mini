from transformers import GPT2LMHeadModel  #standrad gpt-2 architecture out of the box and is compatible with hugging face training tools and moreoevr no weights are loaded just teh blueprint/architecture
from .model_config import get_config

def get_model():
    config = get_config()
    model = GPT2LMHeadModel(config)
    return model
