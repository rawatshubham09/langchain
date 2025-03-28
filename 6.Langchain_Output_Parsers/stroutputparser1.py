from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

#os.environ["HF_HOME"] = "D:/CampusX/huggingface_cache"



model = ChatOpenAI()


#Template 1
template1 = PromptTemplate(template="Write a detailed report on {topic}",
                           input_variables=['topic'])

#Template 2
template2 = PromptTemplate(template="Write a 5 line summary on the following text. /n {text}",
                           input_variables=['text'])


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})


print(result)