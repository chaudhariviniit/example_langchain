from langchain_groq  import ChatGroq
import os
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()
print("API_KEY : ",os.getenv("GROQ_API_KEY"))

MODEL = ChatGroq(model="llama-3.1-8b-instant")

class schema(TypedDict):
    key_theme: Annotated[list[str], "write all keys we discuss in list"]
    summary: Annotated[str, "short and clear summary"]
    sentiment: Annotated[str, "must return sentiment as positive or negative"]
    pros: Annotated[list[str], "provide all pros"]
    cons: Annotated[list[str], "provide all cons"]

prompt =f"""
         The sauce has a pleasant tang, but it sometimes gets overshadowed by the uneven distribution of toppings. When everything comes together in a single bite, you can tell the potential is there—it just needs more consistency. A little more attention to how each slice is assembled would go a long way.
         -Roman Reign
        Texture‑wise, the pizza sits somewhere in the middle. The crust is crisp on the outside yet soft enough inside to feel satisfying, but the topping imbalance makes the overall mouthfeel unpredictable. Some slices feel heavy and overloaded, while others feel like they’re missing that extra something to tie the flavors together.
        -Seth Freakin Rollins
        Despite its flaws, the pizza isn’t a bad choice if you’re craving something quick and comforting. It has a solid foundation, and with a bit more refinement, it could easily stand out. If the makers fine‑tune the topping ratio and improve the presentation, this could become a much more memorable option in the future.  
        -Dean Ambrose  """

structure_model = MODEL.with_structured_output(schema)
response = structure_model.invoke(prompt)
print(response)