from langchain_community.document_loaders import PyPDFLoader  # pip install pypdf
loader = PyPDFLoader("9.RAG_DocumentLoader\\file.pdf")

docs = loader.load()


print(docs[0].page_content)
print(docs[1].metadata)