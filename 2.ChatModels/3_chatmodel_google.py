from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


result = model.invoke("What are 5 state of india?")

print(result.content)