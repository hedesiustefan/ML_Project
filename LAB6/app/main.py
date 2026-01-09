import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_service import answer_question



app = FastAPI()


class ChatRequest(BaseModel):
    user_prompt: str


class ChatResponse(BaseModel):
    chat_response: str
    verdict: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer_question(req.user_prompt)

    if "error" in result:
        return ChatResponse(chat_response=result["error"])

    with open('model_output.txt', 'w') as f:
        f.write(f'Answer: {result["answer"]}')
        f.write(f'Judge Verdict: {result["verdict"]}')

    return ChatResponse(
        chat_response=result["answer"],
        verdict=result["verdict"],
    )
