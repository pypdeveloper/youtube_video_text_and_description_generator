from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
import os
import streamlit as st

st.title("Youtube Video Title And Description Generator")

input = st.text_input(label="Enter your topic here: ")

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0.8, verbose=True)

title_prompt_template = """
Write a creative name for a youtube video based on the topic: {topic}
"""

description_prompt_template = """
Write a creative description for a youtube video based on the topic {topic} and the title {title}
"""

title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["topic"])
title_chain = LLMChain(llm=llm, prompt=title_prompt, verbose=True, output_key="title")

description_prompt = PromptTemplate(template=description_prompt_template, input_variables=['topic', 'title'])
description_chain = LLMChain(llm=llm, prompt=description_prompt, verbose=True, output_key="description")

sequential_chain = SequentialChain(chains=[title_chain, description_chain], input_variables=['topic'], output_variables=["title", "description"], verbose=True)

if input:
    response = sequential_chain({"topic": input})
    st.write(f"Title: {response['title']}")
    st.write(f"Description: {response['description']}")