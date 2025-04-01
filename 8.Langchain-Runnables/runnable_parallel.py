import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"]
)

#chat gpt model
model1 = ChatOpenAI()

# local models
os.environ["HF_HOME"] = "D:/CampusX/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.9, 
                     max_new_tokens=150)
)

model2 = ChatHuggingFace(llm=llm)


parser = StrOutputParser()

# Parallel Runnable
parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model1, parser),
        'linkedin': RunnableSequence(prompt2, model2, parser)
    }
)

result = parallel_chain.invoke({"topic": "AI"})

print(result)