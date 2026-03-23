from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
print("API_KEY :", os.getenv("GROQ_API_KEY"))

model = ChatGroq(model="llama-3.1-8b-instant")

class Review(TypedDict):
    summary: str
    sentiment: str
    features: str
    drawbacks: str

prompt = """
Extract the following fields ONLY:
- summary
- sentiment
- features
- drawbacks

Return them EXACTLY in the schema. No extra fields.

Text:
The crust is great, but the toppings feel a bit overwhelming.
There’s too much cheese in some bites while other parts feel a little empty.
The flavor is decent overall, but it could definitely be more balanced.
Also the presentation looks a bit messy compared to other places.
Hoping they improve the recipe in the next batch. Overall,
it’s an average pizza with some good elements but also a few drawbacks.
"""

structure_output = model.with_structured_output(Review)
response = structure_output.invoke(prompt)
print(response)