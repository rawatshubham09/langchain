from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
#model2 = ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

#Prompts
prompt1 = PromptTemplate(
    template="Generate a Short and simple for the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 question from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and questions into single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes","quiz"]
)

#Output parser
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser,
     "quiz": prompt2 | model1 | parser}
)

merge_chain = prompt3 | model1 | parser

cahin = parallel_chain | merge_chain

text = """LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It provides a standard interface for chains, many integrations with other tools, and end-to-end chains for common applications.

LangChain allows AI developers to develop applications based on the combined Large Language Models (such as GPT-4) with external sources of computation and data. This framework comes with a package for both Python and JavaScript.

Why is LangChain Important?
LangChain helps manage complex workflows, making it easier to integrate LLMs into various applications like chatbots and document analysis. Key benefits include:

Modular Workflow: Simplifies chaining LLMs together for reusable and efficient workflows.
Prompt Management: Offers tools for effective prompt engineering and memory handling.
Ease of Integration: Streamlines the process of building LLM-powered applications.
Key Components of LangChain
1. Chains
Chains define sequences of actions, where each step can involve querying an LLM, manipulating data, or interacting with external tools.

There are two types:

Simple Chains: A single LLM invocation.
Multi-step Chains: Multiple LLMs or actions combined, where each step can take the output from the previous step.
2. Prompt Management
LangChain facilitates managing and customizing prompts passed to the LLM. Developers can use PromptTemplates to define how inputs and outputs are formatted before being passed to the model. It also simplifies tasks like handling dynamic variables and prompt engineering, making it easier to control the LLM’s behavior.

3. Agents
Agents are autonomous systems within LangChain that take actions based on input data. They can call external APIs or query databases dynamically, making decisions based on the situation. These agents leverage LLMs for decision-making, allowing them to respond intelligently to changing input.

4. Vector Database
LangChain integrates with a vector database, which is used to store and search high-dimensional vector representations of data. This is important for performing similarity searches, where the LLM converts a query into a vector and compares it against the vectors in the database to retrieve relevant information.

Vector database plays a key role in tasks like document retrieval, knowledge base integration, or context-based search, providing the model with dynamic, real-time data to enhance responses.

5. Models
LangChain is model-agnostic, meaning it can integrate with different LLMs, such as OpenAI’s GPT, Hugging Face models, DeepSeek R1, and more. This flexibility allows developers to choose the best model for their use case while benefiting from LangChain’s architecture.

6. Memory Management
LangChain supports memory management, allowing the LLM to “remember” context from previous interactions. This is especially useful for creating conversational agents that need context across multiple inputs. The memory allows the model to handle sequential conversations, keeping track of prior exchanges to ensure the system responds appropriately.

"""


result = cahin.invoke({'text': text})

print(result)

cahin.get_graph().print_ascii()