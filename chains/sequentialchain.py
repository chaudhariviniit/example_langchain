from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

templete1 = PromptTemplate(
    template="Write 3 points on this {topic}",
    input_variables=['topic']
)

templete2 = PromptTemplate(\
    template="summarize the points {text} in 2 words"
    )

text_dict = RunnableLambda(lambda para : {'text' : para})

parser = StrOutputParser()

chain = templete1 | model | parser | text_dict | templete2 | model | parser

response = chain.invoke({"topic" : "OS in Linux"})
print(response)