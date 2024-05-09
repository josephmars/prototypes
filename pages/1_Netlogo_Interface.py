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

if 'elements' not in st.session_state:
    st.session_state.elements = []

if 'template' not in st.session_state:
    # Read the template .txt file
    with open('data/netlogo_interface/empty_netlogo.txt', 'r') as file:
        st.session_state.template = file.read()

if 'exp_nlogo' not in st.session_state:
    st.session_state.exp_nlogo = ""
    

# Set page configuration        
st.set_page_config(
    page_title="Create items for NetLogo interface",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

#=======================================================================================================================#

#------------------ Sidebar ------------------#


# Upload your .nlogo file (optional)
st.sidebar.title('1. Upload your .nlogo file (optional)')
nlogo = st.sidebar.file_uploader('Upload your .nlogo file', type='nlogo')
# convert the file to a string
if nlogo is not None:
    st.session_state.template = nlogo.read().decode('utf-8')
    st.session_state.elements = []


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

st.title('Include items for NetLogo interface 📱')

# Read txt file with empty NetLogo interface and store in session state

with open('data/netlogo_interface/empty_netlogo.txt', 'r') as file:
    empty_netlogo = file.read()
st.session_state.empty_netlogo = empty_netlogo



# Call LLM model
if model == 'gpt-3.5-turbo-1106':
    llm = ChatOpenAI(temperature=temp, model=model, seed = seed)
else:
    llm = ChatOpenAI(temperature=temp, model=model)

prompts_list = json.load(open('data/netlogo_interface/interface_dataset.json'))




#------------------ Chat ------------------#
col1, col2 = st.columns(2)
col1.header("AI Chatbot")
col2.header("Included elements")


#------ COL 2 -------#
# Display elements added to the NetLogo interface
if 'elements' in st.session_state and st.session_state.elements:
    for element in st.session_state.elements:
        col2.markdown(f"- {element}")
    
if 'elements' in st.session_state and st.session_state.elements:
    col2.download_button('Download .nlogo', st.session_state.exp_nlogo, file_name='model.nlogo', mime='text/plain') 







#------ COL1 -------#
# initial_prompt = "Enter the items you would like to include in the NetLogo interface"
# st.session_state.messages.append({"role": "assistant", "content": initial_prompt})



# Initialize chat history
if "messages" not in st.session_state:
    initial_prompt = "Enter the items you would like to include in the NetLogo interface"
    st.session_state.messages = [{"role": "assistant", "content": initial_prompt}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with col1.chat_message(message["role"]):
        col1.markdown(message["content"])

# Function to simulate chat interaction
def chat(prompt, answer):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with col1.chat_message("user"):
        col1.markdown(prompt)

    # Display assistant response in chat message container
    with col1.chat_message("assistant"):
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


# Temporal answer
if 'temp_answer' not in st.session_state:
    st.session_state.temp_answer = ""
    
# Accept user input
if user_prompt := col1.chat_input("Which items would you like to include in the NetLogo interface?"):

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Call the chat function with the user prompt and a predefined question
    messages = [SystemMessage(content= task_to_prompt(user_prompt, prompts_list)), HumanMessage(content=user_prompt)]
    AI_prompt = llm(messages).content
    
    messages2 = [SystemMessage(content= second_prompt(AI_prompt, prompts_list)), HumanMessage(content=AI_prompt)]

    answer = llm(messages2).content
    chat(user_prompt, answer)
    # st.session_state.elements.append(answer)
    # col1.success('Item added to model!')
    # Add item to model if button is clicked
    
     
    st.session_state.temp_answer = answer
    
if col1.button('Add to model?') and st.session_state.temp_answer is not "":
    
    col1.success('Item added to model!')
    
    ## Add the item to the nlogo interface
    exp_nlogo = st.session_state.template
    for item in st.session_state.elements:
        exp_nlogo = add_item_to_nlogo_interface(item, exp_nlogo)
    st.session_state.exp_nlogo = exp_nlogo
    
if st.session_state.temp_answer is not "":
    st.session_state.elements.append(st.session_state.temp_answer)
    st.session_state.temp_answer = ""
    


    

        
    



