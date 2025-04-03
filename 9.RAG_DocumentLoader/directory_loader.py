from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="9.RAG_DocumentLoader\\books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Slow as it load all the documents in ram and then process them
"""
docs = loader.load()

for doc in docs:
    print(doc.metadata)
    
"""

# Lazy loader is faster as it loads one page at a time and then process

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)
