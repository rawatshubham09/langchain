import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

dir_path = os.getcwd()
print(dir_path)

cricket_file_path = os.path.join(dir_path,"9.RAG_DocumentLoader","cricket.txt")

loader = TextLoader(cricket_file_path,encoding="utf-8")

docs = loader.load()

print(type(docs))

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Write a summary for the following Poem: \n {poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"poem": docs[0].page_content})

print(result)