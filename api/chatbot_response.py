import os
import sys
import constants
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import APIRouter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
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
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    parser = StrOutputParser()
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know.".

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    loader = TextLoader("data/Research_ Chatbot Content.txt")
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    data = text_splitter.split_documents(text_documents)
    embeddings = OpenAIEmbeddings()
    index_name = "the-musheer-app"
    pinecone = PineconeVectorStore.from_documents(
        data, embeddings, index_name=index_name
    )
    chain = (
        {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    try:
        return chain.invoke(user_query)
    except Exception as e:
        return e