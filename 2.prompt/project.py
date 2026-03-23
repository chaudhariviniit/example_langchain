from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()
print ("API_KEY:" , os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
print("Blog post generator")
print("Provide idea or instruction for post. Type 'exit' to finish.")

topic = input("Enter a specific topic: ")

# Correct: wrap in a list
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a professional blog writer. Make the blog more interesting and look professional."
    ),
    HumanMessagePromptTemplate.from_template(
        "Write a detailed post about {topic}"
    )
])

chat_history = []

while True:
    user_input = input("Idea or instruction or exit: ")
    if user_input.lower() == "exit":
        print("Blog session ended.")
        break

    # Format messages with topic
    messages = prompt_template.format_messages(topic=topic)

    # Add chat history
    messages.extend(chat_history)

    # Add current user input correctly
    messages.append(HumanMessage(content=user_input))

    # Get response
    response = model.invoke(messages)
    print("Post Blog:", response.content)

    # Save conversation history
    chat_history.extend(messages)