from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="""gemini-2.5-flash""")
model2 = ChatGoogleGenerativeAI(model="""gemini-2.5-flash""")

template1 = PromptTemplate(
    template="summarize the the whole site in short and clear with poinst displayy {para}",
    input_variables=['para']
)

template2 = PromptTemplate(
    template="write down 5 question ans based on this{text}",
    input_variables=['text']
)

template3 = PromptTemplate(
    template="Merge both summarize and question & answer: {qa}",
    input_variables=['qa']
)

url="https://www.geeksforgeeks.org/operating-systems/c-scan-disk-scheduling-algorithm/"\

loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

runnable_chain = RunnableParallel(
    summarize = template1 | model | parser,
    qa = template2 | model2 | parser
)

merge_chain = template3 | model2 | parser

origin_chain = runnable_chain | merge_chain

response = origin_chain.invoke(
    {
        "para": docs,
        "text": docs ,
    }
)
print(response)