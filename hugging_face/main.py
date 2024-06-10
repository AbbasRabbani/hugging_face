import os

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._model.to(device)
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        return cls._tokenizer

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Simple Chatbot API"}

@app.post("/predict")
async def predict(message: Message):
    tokenizer = ModelLoader.get_tokenizer()
    model = ModelLoader.get_model()
    
    user_input = message.message
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
