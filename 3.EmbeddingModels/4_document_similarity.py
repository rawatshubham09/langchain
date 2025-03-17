from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embidding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)

documents = [
    "Virat Kohli is an Indian Crickter known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachine Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorker."
    ]

query = "tell me about bowler"

doc_embedding = embidding.embed_documents(documents)
query_embedding = embidding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0] # always in 2d list

print(list(enumerate(scores)))

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(documents[index])
print("similarity score is ",score)
