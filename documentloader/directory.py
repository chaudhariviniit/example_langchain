from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=r"C:\Users\Vinit\OneDrive\Desktop\docs",  # <-- replace with your actual folder
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

# print(len(docs))
# print(docs[0].metadata)

docs1 = loader.lazy_load()

for document in docs1 :
    print(document.metadata)