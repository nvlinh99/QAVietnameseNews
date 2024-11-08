from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.api.controllers.controller import pipeline
from src.api.utils.helper import remove_special_characters

app = FastAPI()

class UserPromptRequest(BaseModel):
    question: str

class AssistantResponse(BaseModel):
    answer: str
    url: List[str]


@app.post("/qa-vn-news", response_model=AssistantResponse)
def get_response(data: UserPromptRequest):
    try:
        print(f"Received question: {data.question}")
        answer, url = pipeline(data.question)
        print(f"Response answer: {answer}")
        print(f"Response url: {url}")
        return AssistantResponse(
            answer=remove_special_characters(answer),
            url=url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    