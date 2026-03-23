from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
print("API_KEY",os.getenv("GROQ_API_KEY"))

model = ChatGroq(model="llama-3.1-8b-instant")
reponse = model.invoke("Hello, define vadodara in one line?")

print(reponse.content)