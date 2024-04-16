 # Career_Chatbot
 Your personal AI based career-counsellor chatbot that answers all your career-related queries. It is specifically trained to cater the needs of Pakistani students.
 
 As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text.

 # To run
 source .venv/bin/activate
 pip install -r requirements.txt
 python RAG_test.py

 # Activate service by entering this into your terminal:
uvicorn main:app --reload

# Add the following prefix to your link on the browser:
/chatbot/docs