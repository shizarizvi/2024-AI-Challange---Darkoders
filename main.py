print("svc_chatbot - starting") 

from fastapi import FastAPI

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.chatbot_response import router as chatbot  

app = FastAPI(openapi_url="/chatbot.json", docs_url="/chatbot/docs")


app.include_router(chatbot, prefix='/chatbot', tags=['chatbot'])