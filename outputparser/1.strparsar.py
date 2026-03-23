from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

templete1 = PromptTemplate(
    template="write a paragraph on {topic}",
    input_variables=['topic']
)

templete2 = PromptTemplate(
    template="Write a 5 line on a {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# Wrap parser output into {"text": ...} for templete2
dict = RunnableLambda(lambda para: {"text": para})

# Correct chain order: parser only after model
chain = templete1 | model | parser | dict | templete2 | model | parser

response = chain.invoke({"topic": "is virat complete his 100 century"})
print(response)