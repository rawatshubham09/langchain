from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence


load_dotenv()

prompt = PromptTemplate(
    template="Write a Joke about this {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke - \n {text}",
    input_variables=["text"]
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "AI"}))