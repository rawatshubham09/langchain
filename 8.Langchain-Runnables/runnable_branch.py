from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=["text"]
)

model = ChatOpenAI()

parser = StrOutputParser()

# Simple report chain
report_generation_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

result = final_chain.invoke({"topic": "Russia-Ukraine war"})

print(result)


# LCEL -> Lang Chain Expression Language (shortcut for RunnableSequence)
# RunnableSequence(prompt1, model, parser)
# we can use -> prompt1 | model | parser

# "|" -> pipe operator