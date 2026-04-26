import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Memory(BaseModel):
    id: str
    title: str
    body: str
    remember: bool
    knowledge_base: str = "General"

class CleanupRequest(BaseModel):
    memories: list[Memory]

class ChatRequest(BaseModel):
    query: str
    personal_memories: list[Memory]
    chat_summary: str
    chat_memories: list[Memory]
    image_data: Optional[str] = None
    recent_messages: list[dict] = []
    power_mode: str = "auto"

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

async def call_claude(user_content: list | str, system: str, model: str, max_tokens: int = 1000):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [
                    {"role": "user", "content": user_content}
                ]
            }
        )
        if response.status_code != 200:
            raise Exception(f"Anthropic API Error {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()

def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured.")

    # 1. Filter personal memories
    active_personal_memories = [m for m in request.personal_memories if m.remember]
    personal_memories_text = "\n".join([f"- {m.title}: {m.body}" for m in active_personal_memories])

    # 2. Format chat memories for Haiku to review
    chat_memories_text = "\n".join([f"ID: {m.id} | Title: {m.title} | Body: {m.body}" for m in request.chat_memories])

    # 3. Format recent messages
    recent_messages_text = ""
    if request.recent_messages:
        recent_3 = request.recent_messages[-3:]
        recent_messages_text = "\n".join([f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in recent_3])

    # 4. Phase 1: Routing & Filtering with Haiku
    haiku_system = (
        "You are the routing and filtering brain of an AI chat app. Your job is to read the user's query and the chat context, "
        "and output a JSON object to decide the next steps. Do not include any text other than the valid JSON."
    )
    
    haiku_prompt = f"""
Context:
[Chat Summary]
{request.chat_summary}

[Personal Memories (Remember=True)]
{personal_memories_text}

[Chat Memories]
{chat_memories_text}

[Recent Messages]
{recent_messages_text}

User Query: {request.query}

Instructions:
1. Identify which Chat Memories are highly relevant to answering the query. List their 'id's in an array.
2. Determine if a web search is needed to answer the query (e.g. for recent events or facts).
3. If search is needed, provide the search term.
4. Determine if the final response requires complex reasoning (high power model) or simple answering (low power model).
5. Extract any new, important personal facts the user shares in their query OR in the Recent Messages (e.g., "my name is X", "I like Y") and format them as new personal memories to save.

Return ONLY a JSON object in this exact format:
{{
  "relevant_chat_memory_ids": ["id1", "id2"],
  "needs_search": true,
  "search_term": "example search",
  "requires_high_power": false,
  "new_personal_memories": [
    {{"title": "Short Title", "body": "Fact details", "remember": true}}
  ]
}}
"""

    base_content = []
    if request.image_data:
        media_type = "image/jpeg"
        base64_data = request.image_data
        if request.image_data.startswith("data:"):
            header, base64_data = request.image_data.split(",", 1)
            media_type = header.split(";")[0].replace("data:", "")
            
        base_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
        })

    haiku_data = {"relevant_chat_memory_ids": [], "needs_search": False, "search_term": "", "requires_high_power": True}
    router_model = "claude-3-7-sonnet-20250219"
    
    try:
        haiku_content = base_content.copy()
        haiku_content.append({"type": "text", "text": haiku_prompt})
        haiku_response = await call_claude(haiku_content, haiku_system, router_model, 300)
        haiku_text = haiku_response["content"][0]["text"]
        haiku_data = json.loads(clean_json(haiku_text))
    except Exception as e:
        print(f"Haiku Parsing Error: {e}")
        # Fallback to high power and no search if JSON parsing fails

    # 5. Filter the relevant chat memories based on Haiku's output
    relevant_ids = haiku_data.get("relevant_chat_memory_ids", [])
    relevant_memories = [m for m in request.chat_memories if m.id in relevant_ids]
    relevant_memories_text = "\n".join([f"- {m.title}: {m.body}" for m in relevant_memories])

    # 6. Search Phase
    search_context = ""
    if haiku_data.get("needs_search") and BRAVE_API_KEY:
        search_term = haiku_data.get("search_term", request.query)
        try:
            async with httpx.AsyncClient() as client:
                brave_response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": search_term},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": BRAVE_API_KEY
                    }
                )
                brave_response.raise_for_status()
                search_data = brave_response.json()

            top_results = search_data.get("web", {}).get("results", [])[:3]
            search_context = "[Web Search Results]\n"
            for result in top_results:
                search_context += f"Title: {result.get('title')}\nDescription: {result.get('description')}\nURL: {result.get('url')}\n\n"
        except Exception as e:
            print(f"Search Error: {e}")
            search_context = "[Web Search Results]\nSearch failed or no results found."

    # 7. Phase 2: Final Response
    final_system = (
        "You are a helpful and intelligent AI assistant. Use the provided context to answer the user's query. "
        "If search results are provided, use them to inform your answer. Keep your response conversational and natural."
    )

    final_prompt = f"""
Context:
[Chat Summary]
{request.chat_summary}

[Personal Memories]
{personal_memories_text}

[Relevant Chat Memories]
{relevant_memories_text}

[Recent Messages]
{recent_messages_text}

{search_context}

User Query: {request.query}
"""

    pm = request.power_mode.lower()
    final_model = "claude-3-7-sonnet-20250219"

    try:
        final_content = base_content.copy()
        final_content.append({"type": "text", "text": final_prompt})
        final_response = await call_claude(final_content, final_system, final_model, 1000)
        final_answer = final_response["content"][0]["text"]
        
        return {
            "response": final_answer,
            "new_personal_memories": haiku_data.get("new_personal_memories", []),
            "debug_info": {
                "haiku_routing": haiku_data,
                "model_used": final_model
            }
        }
    except Exception as e:
        print(f"Final Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the final response: {str(e)}")

@app.post("/cleanup")
async def cleanup_endpoint(request: CleanupRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured.")

    memories_text = "\n".join([f"- Title: {m.title} | Body: {m.body}" for m in request.memories])
    
    system_prompt = (
        "You are an expert librarian and memory consolidation AI. Your task is to organize "
        "a messy list of personal facts and raw chat logs into categorized Knowledge Bases."
        " Do not output any markdown or text other than the raw JSON."
    )
    
    prompt = f"""
Here is a list of unorganized personal memories and chat logs:

{memories_text}

Instructions:
1. Review all the memories and chat logs. Extract the core, permanent facts, preferences, and important context.
2. Merge any duplicate or overlapping information.
3. Rewrite the facts clearly, concisely, and objectively, stripping away any conversational "User said / AI said" formatting.
4. Invent overarching categories (Knowledge Bases) for them (e.g., "Robotics", "Coding", "Preferences", "Worldbuilding"). 
5. Tag every memory with its designated Knowledge Base.

Return ONLY a JSON object in this exact format:
{{
  "cleaned_memories": [
    {{"title": "Short Title", "body": "Consolidated fact details", "knowledge_base": "Category Name", "remember": true}}
  ]
}}
"""
    try:
        response = await call_claude(prompt, system_prompt, "claude-3-7-sonnet-20250219", 2000)
        text = response["content"][0]["text"]
        data = json.loads(clean_json(text))
        return data
    except Exception as e:
        print(f"Cleanup Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while cleaning up memories.")
