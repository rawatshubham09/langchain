from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["HF_HOME"] = "D:/CampusX/huggingface_cache"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


documents = ["Delhi is the capital of India",
             "Deheradun and Gadsain is the capital of Uttarakhand",
             "Paris is the capital of France"]

result = embedding.embed_documents(documents)

print(str(result))