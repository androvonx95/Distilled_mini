from transformers import AutoTokenizer

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")   #GPT-2’s tokenizer is based on Byte Pair Encoding (BPE) which is efficient for both code and natural language
    tokenizer.pad_token = tokenizer.eos_token   #The input tokens get converted into embedding vectors and in order to work on them using matrcies we need to make them the same size ( the larger size ) therefore shorter ones are padded
    return tokenizer
