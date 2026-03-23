from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
print("API_KEY:",os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke("what are you doing? gemini in 3 words")
# print(response.content)
print(response.content)
