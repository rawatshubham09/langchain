from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel


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

joke_chain = RunnableSequence(prompt, model, parser)

parallal_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explain': RunnableSequence(prompt2, model, parser)
    }
)
#chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

# creating final chain
final_chain = RunnableSequence(joke_chain, parallal_chain)

print(final_chain.invoke({"topic": "AI"}))

# Print the graph representation of the chain for visualization
final_chain.get_graph().print_ascii()