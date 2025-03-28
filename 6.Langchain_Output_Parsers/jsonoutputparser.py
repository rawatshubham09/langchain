from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

model = ChatOpenAI()

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the 5 name, age and city of a frictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

"""prompt = template.format()

result = model.invoke(prompt)

final_result = parser.parse(result.content)"""

chain = template | model | parser

final_result = chain.invoke({}) # we dont have any insput so we are sending blank dict

print(final_result)