import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

class QueryRequest(BaseModel):
    query: str

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/aisearch")
async def ai_search(request: QueryRequest):
    if not BRAVE_API_KEY or not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="API keys are not configured.")

    try:
        async with httpx.AsyncClient() as client:
            brave_response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": request.query},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_API_KEY
                }
            )
            brave_response.raise_for_status()
            search_data = brave_response.json()

        top_results = search_data.get("web", {}).get("results", [])[:3]
        search_context = "Web Search Results:\n"
        for result in top_results:
            search_context += f"Title: {result.get('title')}\nDescription: {result.get('description')}\nURL: {result.get('url')}\n\n"

        system_prompt = (
            "You are a helpful AI assistant. Use the provided web search results to answer the user's query. "
            "Keep your response short and sweet, between 1 and 3 sentences."
        )
        user_prompt = f"User Query: {request.query}\n\n{search_context}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            ai_response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 100
                }
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()
            
        final_answer = ai_data["choices"][0]["message"]["content"]
        return {"response": final_answer}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

@app.post("/ai")
async def ai_only(request: QueryRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    try:
        system_prompt = (
            "You are a helpful AI assistant. "
            "Keep your response short and sweet, between 1 and 3 sentences."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            ai_response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.query}
                    ],
                    "max_tokens": 100
                }
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()
            
        final_answer = ai_data["choices"][0]["message"]["content"]
        return {"response": final_answer}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")
