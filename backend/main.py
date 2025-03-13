import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import from langchain_openai instead of langchain.llms
from langchain_openai import ChatOpenAI

# Use PromptTemplate from langchain.prompts
from langchain.prompts import PromptTemplate

# Use RunnableSequence instead of LLMChain
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
if not OPENAI_API_KEY:
    raise ValueError("Please provide OPENAI_API_KEY in your .env file.")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    question: str

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{user_question}")
])


llm = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"
)

chain = template | llm

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running"}

@app.post("/ask")
def ask_question(query: UserQuery):
    user_question = query.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    # Invoke the RunnableSequence with the user_question
    response = chain.invoke({"user_question": user_question})
    return {"answer": response}
