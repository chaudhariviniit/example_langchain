from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(model="llama-3.1-8b-instant")

model2 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template="Generate short note on specific {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write a 5 short question and answer {text}",
    input_variables=['text']
)


prompt3 = PromptTemplate(
    template="Merge notes and question and ans into single unit {notes} , {qa}",
    input_variables=['notes','qa']
)

parser = StrOutputParser()
runnable_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser   ,
    'qa' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = runnable_chain | merge_chain

text = """
India’s history spans thousands of years, beginning with the Indus Valley Civilization and evolving through powerful empires, colonial rule, and independence in 1947. It is a story of cultural richness, political struggles, and social transformations that shaped modern India.
The earliest known civilization in India was the Indus Valley Civilization (2600–1900 BCE), centered around cities like Harappa and Mohenjo-Daro. It was marked by advanced urban planning, trade networks, and craft specialization. After its decline, the Vedic Age (1500–500 BCE) saw the rise of Aryan culture, the composition of the Vedas, and the foundations of Hinduism. Later, the Maurya Empire (321–185 BCE) under Emperor Ashoka unified much of India and spread Buddhism across Asia.
From the 8th century onward, Islamic influence entered India, leading to the establishment of powerful dynasties such as the Delhi Sultanate (1206–1526). This period saw the blending of Hindu and Islamic cultures, producing rich traditions in art, architecture, and literature. The Mughal Empire (1526–1857) brought centralized administration, monumental architecture like the Taj Mahal, and flourishing trade. Akbar’s reign in particular is remembered for religious tolerance and cultural synthesis.
By the 18th century, the decline of the Mughals allowed the British East India Company to expand its control. After the Revolt of 1857, India came under direct British Crown rule, known as the British Raj (1858–1947). This period saw economic exploitation, famines, and social reforms, but also the rise of nationalist movements. Leaders like Mahatma Gandhi, Jawaharlal Nehru, and Subhas Chandra Bose spearheaded struggles for independence through nonviolent resistance and mass mobilization.
"""

response = final_chain.invoke(text)

print(response)