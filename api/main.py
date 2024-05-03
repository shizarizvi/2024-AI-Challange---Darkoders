import os
from dotenv import load_dotenv
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

def chatbot(user_query):
    # Loads the document, split it into chunks, embed each chunk and load it into the vector store.
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    parser = StrOutputParser()
    template = """
    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    loader = TextLoader("data/musheer_data.txt")
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
    

print("\n\n **Welcome to the Musheer Chatbot. Your personal career counsellor.**")
while True:
    user_query=input("\nYou: ")
    if user_query in ['exit', 'quit', 'end', 'stop', 'close']:
        break 
    response = chatbot(user_query)
    print(f'\nMusheer: {response}')