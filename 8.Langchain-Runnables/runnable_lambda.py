from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda


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

def word_count(text: str) -> int:
    return len(text.split())

joke_chain = RunnableSequence(prompt, model, parser)

parallal_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explain': RunnableSequence(prompt2, model, parser),
        'word_count': RunnableLambda(word_count)

    }
)
#chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

# creating final chain
final_chain = RunnableSequence(joke_chain, parallal_chain)

print(final_chain.invoke({"topic": "School"}))

# Print the graph representation of the chain for visualization
final_chain.get_graph().print_ascii()