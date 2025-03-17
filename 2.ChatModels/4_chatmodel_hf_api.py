from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                          task="text-generation")

model = ChatHuggingFace(llm=llm)

result = model.invoke("Name any 5 state of India ?")


print(result.content)