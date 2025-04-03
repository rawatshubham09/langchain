from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()

chain = prompt | model | parser


url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"

loader = WebBaseLoader(url)

docs = loader.load()


result = chain.invoke({"question": "Tell me about this product?",
              "text": docs[0]})

print(result)
#print(len(docs))

#print(docs[0])