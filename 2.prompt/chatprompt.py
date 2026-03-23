from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Do you want to know about {subject}"),
    HumanMessagePromptTemplate.from_template("yes tell me something curious about {subject}")  ])

prompt = chat_prompt.format_messages(subject = "Derived Maths")
print(prompt)
