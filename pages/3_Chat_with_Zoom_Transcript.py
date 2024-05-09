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

from modeler_agent import *

#=======================================================================================================================#
load_dotenv()

# Initialize session state

if 'data' not in st.session_state:
    st.session_state.data = None

if 'agent' not in st.session_state:
    st.session_state.agent = None

st.set_page_config(
    page_title="Chat with Zoom Transcript",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

#=======================================================================================================================#

#------------------ Sidebar ------------------#


# Upload your .nlogo file (optional)
st.sidebar.title('1. Upload your .WEBVTT file')
trans = st.sidebar.file_uploader('Upload your .WEBVTT file', type='vtt')
if trans is not None:
    transcript = trans.read().decode('utf-8')   


st.sidebar.title('2. Select Generative Model')

# Select model from GPT-3.5 or GPT-4
model = st.sidebar.selectbox('Select model', ['gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0125-preview'])
if model == 'gpt-3.5-turbo-1106':
    seed = 0 
    st.sidebar.warning(f'Reproducibility option enabled')
    
# Select temperature for the model
temp = st.sidebar.slider("Select temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05, disabled=False, label_visibility="visible")

#=======================================================================================================================#

#------------------ Main Page ------------------#

st.title('Chat with Zoom Recording')

if transcript is not None:
    t = ""
    # Get  the text from each timestamp
    for i in transcript.split("\n")[2:-2]:
        try:
            if type(int(i)) == int:
                t += "\n"
        except:
            pass
        t += i + "\n"
        

    
    complete = ""
    for timestap in str(t).split('\n\n')[2:-2]:
        obj = timestap.replace('Hector M. Garcia: ', ' ').split('\n')
        start_time = obj[1].split(' --> ')[0]
        end_time = obj[1].split(' --> ')[1]
        
        if (int(obj[0])-1) % 5 == 0:
            complete = complete.replace('%%%ENDTIME%%%', end_time)
            complete += f"# {start_time}  --> %%%ENDTIME%%%\n"
            
        complete += f"{obj[2]}\n\n" 
        
    
    from langchain.text_splitter import MarkdownHeaderTextSplitter


    headers_to_split_on = [
        ("#", "Time")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(complete)

    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain_openai import ChatOpenAI

    # Call LLM model
    if model == 'gpt-3.5-turbo-1106':
        llm = ChatOpenAI(temperature=temp, model=model, seed = seed)
    else:
        llm = ChatOpenAI(temperature=temp, model=model)



    embedding = OpenAIEmbeddings()
    db = Chroma.from_documents(md_header_splits, embedding)

    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )



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
if user_prompt := st.chat_input("What do you want to ask about the transcript?"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Call the chat function with the user prompt and a predefined question

    answer = qa_chain({"query": user_prompt})["result"]
    chat(user_prompt, answer)



