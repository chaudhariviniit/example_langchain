from langchain_core.prompts import PromptTemplate

dynamic_prompts = PromptTemplate(
    template="We discuss on {topic} and use {method} to make result",
    input_variables=["topic","method"]
)

dynamic_text = dynamic_prompts.format(topic = "AI-agents",method = "Google-Meet")
print(dynamic_text)