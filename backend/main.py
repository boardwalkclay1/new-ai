# backend/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list[dict]  # [{ "role": "user" | "assistant" | "system", "content": "..." }]

class ChatResponse(BaseModel):
    reply: str

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

SYSTEM_PROMPT = "You are a helpful, precise open-source AI assistant."

def build_prompt(messages: list[dict]) -> str:
    parts = [f"<s>[SYSTEM] {SYSTEM_PROMPT}"]
    for m in messages:
        role = m.get("role", "user").lower()
        content = m.get("content", "")
        if role == "user":
            parts.append(f"\n[USER] {content}")
        elif role == "assistant":
            parts.append(f"\n[ASSISTANT] {content}")
        elif role == "system":
            parts.append(f"\n[SYSTEM] {content}")
    parts.append("\n[ASSISTANT]")
    return "".join(parts)

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = build_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "[ASSISTANT]" in full_text:
        reply = full_text.split("[ASSISTANT]", 1)[1].strip()
    else:
        reply = full_text.strip()
    return ChatResponse(reply=reply)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
