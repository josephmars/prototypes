import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv


#=======================================================================================================================#
load_dotenv()

# Set page configuration        
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

#=======================================================================================================================#

#------------------ Sidebar ------------------#



# Upload your .pdf file 
st.sidebar.title('Upload your .pdf file (optional)')
pdf = st.sidebar.file_uploader('Upload your .pdf file', type='pdf')




#=======================================================================================================================#

#------------------ Main Page ------------------#

st.title('Chat with PDF')

col1, col2 = st.columns(2)


if pdf is not None:
    # Read the PDF file
    pdf_reader = PdfReader(pdf)
    # Extract the content
    content = ""
    for page in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[page].extract_text()
    
    # Subtitle
    col1.subheader('Content of the PDF file')
    # Display the content
    edited_content = col1.text_area('Your document (Editable in case the format was not right):', height=600, value=content) 
    
    
    from langchain_openai  import ChatOpenAI, OpenAIEmbeddings 
    import sys
    # Add the directory containing the .py module to the sys.path
    sys.path.append('./../')
    from agents.RAGAgent import RAGAgent

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0613')

    embeddings = OpenAIEmbeddings()
    #  Load and split documents
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.text_splitter import MarkdownHeaderTextSplitter

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)

    splits = text_splitter.split_text(edited_content)

    # join content in markdown format (each split is a header)
    joined_text = ""
    for i in range(len(splits)):
        joined_text += f"# {i+1}\n{splits[i]}\n"

    headers_to_split_on = [("#", "Page")]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(joined_text)
    template = """Use the following pieces of context from the document to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context:
        {context}
    """

    # Create the RAG agent
    rag_agent = RAGAgent(llm, md_header_splits, embeddings, template)
    

    col2.subheader('What is your question?')
    question = col2.text_area('Enter question:', height=100, value = "What is the document about?") 
    

    response = rag_agent.chat(question, return_source_documents=True)
    col2.subheader('Answer:')
    col2.write(response["answer"])
    
    col2.subheader('Source documents:')
    for doc in response["source_documents"]:
        col2.write("- "+doc.page_content)
    
