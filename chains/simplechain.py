from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template="generate 3 lines of this {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({'topic' : 'Chemistry'})

print(response)