import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import the RAG logic from our new file ---
from rag_logic import create_rag_chain

# --- 1. SET UP THE SERVER & LOAD THE RAG CHAIN ---

# This dictionary will hold our "global" RAG chain
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the server starts
    print("Server starting up...")
    print("Loading RAG chain...")
    
    try:
        # Call the function from rag_logic.py to build the chain
        app_state["rag_chain"] = create_rag_chain()
        print("RAG chain setup complete. Server is ready.")
    except Exception as e:
        print(f"Error during RAG chain setup: {e}")
        print("Server will start, but /chat endpoint will return an error.")
        app_state["rag_chain"] = None
    
    yield
    
    # This code runs when the server shuts down
    print("Server shutting down...")
    app_state.clear()


# --- 2. CREATE THE FASTAPI APP ---
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
# This is CRITICAL to allow your HTML file (on a different "origin")
# to talk to this Python server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for simplicity)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 3. DEFINE THE API ENDPOINTS ---

# Pydantic models for request and response
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "LangChain RAG API is running. POST to /chat"}

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    This is the main chat endpoint.
    It receives a question and uses the RAG chain to get an answer.
    """
    rag_chain = app_state.get("rag_chain")
    
    if not rag_chain:
        # This will happen if GROQ_API_KEY was missing or another error occurred
        return ChatResponse(answer="Error: RAG chain is not initialized. Check server logs.")

    print(f"Received question: {request.question}")
    
    # Run the chain (just like your original script)
    response_text = rag_chain.invoke(request.question)
    
    print(f"Sending answer: {response_text}")
    return ChatResponse(answer=response_text)

# --- 4. (Optional) Run the server if this file is executed directly ---
if __name__ == "__main__":
    import uvicorn
    # Make sure to set your GROQ_API_KEY environment variable before running
    # Example: export GROQ_API_KEY='your_key_here'
    print("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)