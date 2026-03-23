from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field,BaseModel
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class schema(BaseModel):
    name: str = Field(description="write the nsme of a any random guy from declare place")
    age: int = Field(description="written age of a person")
    city: str = Field(description="desc city in 3 word , where person live")

parser = PydanticOutputParser(pydantic_object=schema)

template1 = PromptTemplate(
    template="write name,age and city name of fictional {place} person\n"
                "written in following describe format\n"
                "{format_instruction}\n",
            input_variables=['place'],
            partial_variables={"format_instruction" : parser.get_format_instructions()}
)

chain = template1 | model | parser
result = chain.invoke({'place' : 'us'})
print(result)