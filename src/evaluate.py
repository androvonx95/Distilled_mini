import torch
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_model
from src.tokenizer import get_tokenizer
from safetensors.torch import load_file


def load_model():
    model = get_model()
    checkpoint_dir = "checkpoints"
    latest_checkpoint = max([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint")], key=lambda x: int(x.split("-")[-1]))
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

def chat():
    print("\nChat with the model! Type 'exit' to quit.")
    model, device = load_model()
    tokenizer = get_tokenizer()
    
    # Start with a friendly greeting
    print("\nAssistant: Hello! I'm here to help. How can I assist you today?\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("\nGoodbye! Have a great day!")
            break
            
        try:
            # Format the input as a friendly conversation
            prompt = f"You: {user_input}\nAssistant:"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    min_length=30,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.85,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_beams=1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            
            # Clean up the response
            response = response.replace("\n", " ").strip()
            
            # If response is empty or too short, try again
            if len(response) < 10:
                response = "I'm not sure I understand. Could you please rephrase your question?"
            
            print(f"\nAssistant: {response}\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    chat()
