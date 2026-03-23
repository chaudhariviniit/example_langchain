from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from typing import Literal
from pydantic import  BaseModel,Field
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()
model1 = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive','Negative'] = Field(description="The sentiment for the feedback must be either positive or negative")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="write a clear sentiment 'Postive or 'Negative' on feedback"
    "write in json object thing with field 'sentiment' \n"
    "\n feedback = {feedback}"
    "\n\n(format_instruction)",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

main_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template="Write a appropriate sentiment as postive on {feedback}",
    input_variables=['feedback']
)


prompt3 = PromptTemplate(
    template="Write a appropriate sentiment as negative on {feedback}",
    input_variables=['feedback']
)

branch = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model1 | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model1 | parser),
     parser 
)


chain = main_chain | branch
response = chain.invoke({"feedback":"*I hadnot an amazing experience at the concert last night! The band wasnot incredible and the energy of the crowd wasnt even electric."})
print(response)