from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import Literal,TypedDict

load_dotenv()
print("API_KEY",os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

json_schema= {
    "title" : "review",
    "type" : "object",
    "properties" : {
        "key_theme" : { "type" : "array",
                       "items":{"type" : "string"},
                       "descryption" : "write down all the key theme that we discussed"
                       },
        
        "summary" : { 
            "type" : "string",
        "descryption" : "write down short and clear summary that we discussed"
        },
        "sentiment" : { "type" : "array",
                       "enum" : ["positive","negative"],
        "description" : "write down proper seentiment"
          },
        "pros" : { "type" : "array",
                    "items":{"type" : "string"},
                    "descryption" : "write down all the pros that we discussed"
                       },
        "cons" : { "type" : "array",
                    "items":{"type" : "string"},
                    "descryption" : "write down all cons that we discussed"
                       },
        "name" : { "type" : "array",
                    "items":{"type" : "string"},
                    "descryption" : "write down all the name of reviewrs that we discussed"
                       }

}        
}

prompt = """
        The sauce has a pleasant tang, but it sometimes gets overshadowed by the uneven distribution of toppings. When everything comes together in a single bite, you can tell the potential is there—it just needs more consistency. A little more attention to how each slice is assembled would go a long way.
         -Roman Reign
        Texture‑wise, the pizza sits somewhere in the middle. The crust is crisp on the outside yet soft enough inside to feel satisfying, but the topping imbalance makes the overall mouthfeel unpredictable. Some slices feel heavy and overloaded, while others feel like they’re missing that extra something to tie the flavors together.
        -Seth Freakin Rollins
        Despite its flaws, the pizza isn’t a bad choice if you’re craving something quick and comforting. It has a solid foundation, and with a bit more refinement, it could easily stand out. If the makers fine‑tune the topping ratio and improve the presentation, this could become a much more memorable option in the future.  
        -Dean Ambrose 
        The flavor profile is dependable, leaning on familiar comfort rather than bold experimentation. That said, the lack of balance in toppings sometimes makes the pizza feel like it’s holding back from being truly great. A more deliberate layering of ingredients could elevate it from “good enough” to something worth recommending every time.
        -CM Punk
        Visually, the pizza doesn’t always impress—some slices look inviting, while others appear uneven and rushed. Presentation matters, and with a bit more care in how each slice is arranged, the overall experience could match the promise of its taste. It’s close to being a standout option, but consistency in both look and flavor will determine whether it earns repeat visits.
        -Brock Lesner """

structure_model = model.with_structured_output(json_schema,strict = True)
response  = structure_model.invoke(prompt)

print(response)