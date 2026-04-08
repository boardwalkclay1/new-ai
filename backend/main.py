import os
import uuid
import json
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

# ---------- CONFIG ----------

MODEL_CONFIGS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3": "meta-llama/Llama-3-8B-Instruct",
    "gemma": "google/gemma-7b-it",
}

DEFAULT_MODEL = "mistral"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FILES_DIR = "./boardwalk_files"
os.makedirs(FILES_DIR, exist_ok=True)

# ---------- FASTAPI APP ----------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DATA MODELS ----------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "auto"   # "auto", "mistral", "llama3", "gemma"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    used_models: List[str]
    session_id: str

# ---------- GLOBALS: MODELS, MEMORY, RAG ----------

MODELS: Dict[str, tuple] = {}
SYSTEM_PROMPT = "You are Boardwalk AI, a precise, tool-using open-source assistant."

# simple in-memory conversation memory per session
SESSION_MEMORY: Dict[str, List[Dict]] = {}

# vector DB for RAG
chroma_client = chromadb.Client()
rag_collection = chroma_client.get_or_create_collection("boardwalk_rag")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ---------- MODEL LOADING / ROUTING ----------

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

def choose_model(messages: List[Message], requested: str) -> str:
    if requested and requested != "auto":
        return requested if requested in MODEL_CONFIGS else DEFAULT_MODEL

    # crude router: if user mentions "code" or "debug", prefer llama3
    last_user = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user = m.content.lower()
            break

    if any(k in last_user for k in ["code", "bug", "error", "stack trace", "function"]):
        return "llama3"
    if any(k in last_user for k in ["story", "poem", "creative"]):
        return "gemma"
    return DEFAULT_MODEL

def build_prompt(messages: List[Message], tools_desc: str, rag_context: str) -> str:
    parts = [f"<s>[SYSTEM] {SYSTEM_PROMPT}\n\n[TOOLS]\n{tools_desc}\n\n[RAG]\n{rag_context}"]
    for m in messages:
        role = m.role.lower()
        if role == "user":
            parts.append(f"\n[USER] {m.content}")
        elif role == "assistant":
            parts.append(f"\n[ASSISTANT] {m.content}")
        elif role == "system":
            parts.append(f"\n[SYSTEM] {m.content}")
    parts.append("\n[ASSISTANT]")
    return "".join(parts)

def generate_with_model(model_key: str, messages: List[Message], tools_desc: str, rag_context: str) -> str:
    tokenizer, model = load_model(model_key)
    prompt = build_prompt(messages, tools_desc, rag_context)
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

# ---------- TOOLS (INCLUDING REAL FILE WRITE) ----------

def tool_write_file(path: str, content: str) -> str:
    safe_name = path.replace("..", "_").strip().lstrip("/")
    full_path = os.path.join(FILES_DIR, safe_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File written to {full_path}"

def tool_list_files() -> str:
    files = []
    for root, _, fs in os.walk(FILES_DIR):
        for name in fs:
            rel = os.path.relpath(os.path.join(root, name), FILES_DIR)
            files.append(rel)
    return json.dumps(files, indent=2)

TOOLS = {
    "write_file": {
        "description": "Write a text file into the Boardwalk AI workspace.",
        "schema": {"path": "string", "content": "string"},
        "func": tool_write_file,
    },
    "list_files": {
        "description": "List files previously written by Boardwalk AI.",
        "schema": {},
        "func": lambda: tool_list_files(),
    },
}

def tools_description() -> str:
    desc = []
    for name, meta in TOOLS.items():
        desc.append(f"- {name}: {meta['description']}")
    return "\n".join(desc)

def maybe_parse_tool_call(text: str):
    """
    Very simple protocol:
    If the assistant outputs a line like:
    <tool:write_file>{"path": "notes.txt", "content": "hello"}
    we detect and execute it once, then return the tool result.
    """
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("<tool:") and ">" in line:
            header, payload = line.split(">", 1)
            tool_name = header.replace("<tool:", "").strip()
            payload = payload.strip()
            try:
                data = json.loads(payload or "{}")
            except Exception:
                data = {}
            return tool_name, data
    return None, None

def execute_tool(tool_name: str, args: dict) -> str:
    if tool_name not in TOOLS:
        return f"Tool '{tool_name}' not found."
    func = TOOLS[tool_name]["func"]
    try:
        if args:
            return func(**args)
        return func()
    except TypeError:
        return "Invalid arguments for tool."
    except Exception as e:
        return f"Tool error: {e}"

# ---------- RAG HELPERS ----------

def rag_add_document(doc_id: str, text: str, metadata: dict):
    embedding = embed_model.encode([text])[0].tolist()
    rag_collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[text],
    )

def rag_query(query: str, k: int = 3) -> str:
    if not query.strip():
        return ""
    embedding = embed_model.encode([query])[0].tolist()
    res = rag_collection.query(
        query_embeddings=[embedding],
        n_results=k,
    )
    docs = res.get("documents", [[]])[0]
    if not docs:
        return ""
    joined = "\n\n".join(docs)
    return joined

# ---------- MEMORY HELPERS ----------

def get_session_id(existing: Optional[str]) -> str:
    return existing or str(uuid.uuid4())

def append_memory(session_id: str, role: str, content: str):
    SESSION_MEMORY.setdefault(session_id, [])
    SESSION_MEMORY[session_id].append({"role": role, "content": content})

def get_memory_messages(session_id: str) -> List[Message]:
    raw = SESSION_MEMORY.get(session_id, [])
    return [Message(role=m["role"], content=m["content"]) for m in raw]

# ---------- API ENDPOINTS ----------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = get_session_id(req.session_id)

    # merge incoming messages with stored memory
    history = get_memory_messages(session_id)
    all_messages = history + req.messages

    # RAG context from last user message
    last_user = ""
    for m in reversed(all_messages):
        if m.role == "user":
            last_user = m.content
            break
    rag_ctx = rag_query(last_user)

    # choose model
    model_key = choose_model(req.messages, (req.model or "auto").lower())
    reply = generate_with_model(model_key, all_messages, tools_description(), rag_ctx)

    # check for tool call
    tool_name, tool_args = maybe_parse_tool_call(reply)
    if tool_name:
        tool_result = execute_tool(tool_name, tool_args or {})
        # append tool result as assistant message
        reply = f"(Tool {tool_name} result)\n{tool_result}"

    # update memory
    for m in req.messages:
        append_memory(session_id, m.role, m.content)
    append_memory(session_id, "assistant", reply)

    return ChatResponse(
        reply=reply,
        used_models=[model_key],
        session_id=session_id,
    )

@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...), namespace: str = Form("default")):
    text = (await file.read()).decode("utf-8", errors="ignore")
    doc_id = f"{namespace}:{file.filename}"
    rag_add_document(doc_id, text, {"namespace": namespace, "filename": file.filename})
    return {"status": "ok", "doc_id": doc_id}

@app.get("/api/files")
def list_files_api():
    return {"files": json.loads(tool_list_files())}

@app.post("/api/files/write")
def write_file_api(path: str = Form(...), content: str = Form(...)):
    msg = tool_write_file(path, content)
    return {"status": "ok", "message": msg}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
