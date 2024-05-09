import streamlit as st
import pandas as pd
import os
import json

from langchain_community.llms import OpenAI
import random
import time
import tempfile
from langchain.agents import load_tools 
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

from agents.modeler_agent import *

#=======================================================================================================================#
load_dotenv()

# Initialize session state

if 'data' not in st.session_state:
    st.session_state.data = None

if 'agent' not in st.session_state:
    st.session_state.agent = None

st.set_page_config(
    page_title="Coding with NetLogo",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded",
)

#=======================================================================================================================#

#------------------ Sidebar ------------------#



st.sidebar.title('Select Generative Model')

# Select model from GPT-3.5 or GPT-4
model = st.sidebar.selectbox('Select model', ['gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0125-preview'])
if model == 'gpt-3.5-turbo-1106':
    seed = 0 
    st.sidebar.warning(f'Reproducibility option enabled')
    
# Select temperature for the model
temp = st.sidebar.slider("Select temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05, disabled=False, label_visibility="visible")

#=======================================================================================================================#

#------------------ Main Page ------------------#

st.title('Coding with NetLogo 🐢')


# Call LLM model
if model == 'gpt-3.5-turbo-1106':
    llm = ChatOpenAI(temperature=temp, model=model, seed = seed)
else:
    llm = ChatOpenAI(temperature=temp, model=model)


# Add the directory containing the .py module to the sys.path
import sys
sys.path.append('./../')

# Load the OpenAI embeddings
from langchain_openai  import ChatOpenAI, OpenAIEmbeddings 
from agents.RAGAgent import RAGAgent

embeddings = OpenAIEmbeddings() 
#  Load and split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the manual
loader = PyPDFLoader("../data/manual/manual.pdf")
pages = loader.load()

# Join all pages into a single string
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)

splits = text_splitter.split_text(joined_page_text)

# join content in markdown format (each split is a header)
joined_text = ""
for i in range(len(splits)):
    joined_text += f"# {i+1}\n{splits[i]}\n"

headers_to_split_on = [("#", "Page")]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
md_header_splits = markdown_splitter.split_text(joined_text)






#------------------ Chat ------------------#


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to simulate chat interaction
def chat(prompt, answer):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = answer  # Replace this with the actual assistant response logic

        
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.03)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

#------------------ Chat ------------------#

# Accept user input
if user_prompt := st.chat_input("What would you like to code in NetLogo? 🐢"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Call the chat function with the user prompt
    
    
    
    template = """Use the following pieces of context from the NetLogo Manual to generate code for the request at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not provide explanations or descriptions, just the code.
    Request: {question}
    Context from NetLogo manual:
    {context}
"""
    examples = """
to check-outline
  ;; ensures agents retain their outline when changing breed
  if staying-at-home? [
    set shape "person-outline"
  ]
end

to set-breed-susceptible
  set breed susceptibles
  set p-infect (p-infect-base / 100)
  set to-become-exposed? false
  set asked-to-isolate? false
  if visual-elements? [
    set color green
    check-outline
  ]
end

to set-breed-exposed
  set breed exposeds
  set incubation-countdown (log-normal incubation-mean incubation-stdev 0)
  set to-become-asymptomatic? false
  set contact-list []
  set tested? false
  set contacts-alerted? false
  set presym-period (random 3 + 1)
  if visual-elements? [
    set color yellow
    check-outline
  ]
end

to set-breed-asymptomatic
  set breed asymptomatics
  check-symptoms               ;; sets will-develop-sym? and countdown
  set to-become-sym? false
  set to-recover? false
  ;; contact-list carries over from exposeds
  ;; tested? carries over from exposeds
  if visual-elements? [
    set color violet
    check-outline
  ]
end

to check-symptoms
  ;; check whether the agent will remain asymptomatic or develop symptoms
  ;; and assign a value to the countdown accordingly
  let p random-float 100
  ;; adjust asymptomatic prevalence based on age
  let asym-prevalence (actual-asym-prevalence age)
  ifelse p < asym-prevalence [
    set will-develop-sym? false
    set countdown (normal-dist recovery-mean recovery-stdev) ;; recovery countdown
  ] [ ;; else
    set will-develop-sym? true
    set countdown presym-period                              ;; symptoms countdown
  ]
end

to set-breed-symptomatic
  set breed symptomatics
  check-death                  ;; sets will-die? and countdown
  set to-die? false
  set to-recover? false
  ;; contact-list carries over from asymptomatic
  ;; tested? carries over from asymptomatic
  if visual-elements? [
    set color red
    check-outline
  ]
end

to check-death
  ;; check whether the agent will recover or die
  ;; and assign a value to the countdown accordingly
  let p random-float 100
  ifelse p <= p-death [
    set will-die? true
    set countdown (normal-dist death-mean death-stdev) ;; death countdown
  ] [ ;; else
    set will-die? false
    set countdown (normal-dist recovery-mean recovery-stdev) ;; recovery countdown
  ]
end

to set-breed-recovered
  set breed recovereds
  if lose-immunity? [
    set immunity-countdown (log-normal 1 0.5 min-immunity-duration)
    ]
  set to-become-susceptible? false
  if visual-elements? [
    set color grey
    check-outline
  ]
end
"""
    # Create the RAG agent
    rag_agent = RAGAgent(llm, md_header_splits, embeddings, prompt = template)
    # Chat with RAG agent
    answer, source_docs = rag_agent.chat(user_prompt, return_source_documents=True)
    output = f"{answer}\n\nSource Documents:\n{[source_docs[i].metadata for i in range(len(source_docs))]}"
    chat(user_prompt, output)



