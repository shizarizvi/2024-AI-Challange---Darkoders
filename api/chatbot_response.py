import os
import sys
import constants
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Form, Query
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class Request(BaseModel):
    user_query: str 
    
router = APIRouter()

@router.get("/")
async def health():
    print("chatbot.api.health")
    return {"message": "svc_chatbot is alive"}

@router.post("/get_chatbot_response")
async def create(request: Request):
    user_query = request.user_query
    chatbot_response = chatbot(user_query)
    return {"chatbot_response": chatbot_response}


def chatbot(user_query):
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    data = TextLoader('data.txt').load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_data = text_splitter.split_documents(data)
    db = Chroma.from_documents(split_data, OpenAIEmbeddings())

    query = user_query
    response = db.similarity_search(query)
    return response[0].page_content

