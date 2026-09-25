from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
