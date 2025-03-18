from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatOpenAI()


st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name",["Attention is all you need", "BERT: pre-training of Deep Bidirectional Transformers",
                                                         "GPT-3: Language Models are Few-Shot Learners", 
                                                         "RoBERTa: A Robustly Optimized BERT Approach",
                                                         "The Unreasonable Effectiveness of Recurrent Neural Networks"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical",
                                                        "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraph)", "Medium (3-4 paragraph)", "Long (4-5 paragraph)"])

template = load_prompt("template.json")

prompt = template.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)