from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate


load_dotenv()

#os.environ["HF_HOME"] = "D:/CampusX/huggingface_cache"


llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it",
                          task="text-generation")

model = ChatHuggingFace(llm = llm)


#Template 1
template1 = PromptTemplate(template="Write a detailed report on {topic}",
                           input_variables=['topic'])

#Template 2
template2 = PromptTemplate(template="Write a 5 line summary on the following text. /n {text}",
                           input_variables=['text'])

prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result1 = model.invoke(prompt2)

print(result1.content)