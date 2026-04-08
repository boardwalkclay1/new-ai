# backend/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
    model: str | None = "ensemble"  # "mistral", "llama3", "gemma", or "ensemble"

class ChatResponse(BaseModel):
    reply: str
    used_models: list[str]

SYSTEM_PROMPT = "You are a helpful, precise open-source AI assistant."

MODEL_CONFIGS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3": "meta-llama/Llama-3-8B-Instruct",
    "gemma": "google/gemma-7b-it",
}

MODELS = {}

def load_model(key: str):
    if key in MODELS:
        return MODELS[key]
    name = MODEL_CONFIGS[key]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    MODELS[key] = (tokenizer, model)
    return MODELS[key]

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

def generate_with_model(model_key: str, messages: list[dict]) -> str:
    tokenizer, model = load_model(model_key)
    prompt = build_prompt(messages)
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
    return reply

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    target = (req.model or "ensemble").lower()

    if target == "ensemble":
        used = []
        replies = []
        for key in MODEL_CONFIGS.keys():
            try:
                r = generate_with_model(key, req.messages)
                replies.append(f"[{key}] {r}")
                used.append(key)
            except Exception:
                continue
        if not replies:
            return ChatResponse(reply="All models failed to respond.", used_models=[])
        merged = "\n\n".join(replies)
        return ChatResponse(reply=merged, used_models=used)

    if target not in MODEL_CONFIGS:
        target = "mistral"

    reply = generate_with_model(target, req.messages)
    return ChatResponse(reply=reply, used_models=[target])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
