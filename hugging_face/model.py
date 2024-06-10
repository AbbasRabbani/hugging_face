'''import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Hugging Face token from environment variables
# token = os.getenv("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")
#
# Load the tokenizer and model with the token
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
#model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

#tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#model = AutoModelForCausalLM.from_pretrained("distilgpt2")

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_response(user_input):
    # Tokenize the input text and move to GPU if available

    
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    # Generate a response
    outputs = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    '''