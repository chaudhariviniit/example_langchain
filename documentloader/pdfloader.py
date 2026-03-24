from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Twsl_[dsa].pdf")

docs = loader.load()

print(docs[0].metadata)