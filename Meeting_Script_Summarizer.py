import streamlit as st
import os
from io import BytesIO
import docx2txt  # Import the docx2txt library

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def generate_minutes(text_data, system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease summarize the following text for a meeting minute: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .docx file to generate minutes of the meeting.")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"])

    with st.sidebar:
    # Chat option
     user_question = st.text_input("Ask a question about the document:")
     if uploaded_file is not None:
        submit_buttonn = st.button('Submit')
     else:
        submit_buttonn = None

    # Enable button only if file is uploaded
    if uploaded_file is not None:
        submit_button = st.button('Generate Output')
    else:
        submit_button = None

    if submit_button:
        # Read the uploaded Word document
        docx_data = BytesIO(uploaded_file.read())
        text_data = docx2txt.process(docx_data)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )

        texts = text_splitter.split_text(text_data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vector_store = FAISS.from_texts(texts, embeddings)

        # Access system message content
        system_message_content = "ou are an AI assistant tasked with generating Meeting Minutes (MOM) from the provided text, including details such as Participants, Date, and Time and Summary and conclusion in different section in bullet points.Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points.Please ensure to follow the specified sequence: 1)Participants, 2) Date & Time, 3) Brief Discussion/Agenda , 4) Summary of Discussion, 5) Meeting Conclusion" 

        result = generate_minutes(text_data, system_message_content, vector_store)
        st.write("**Output:**")
        st.write(result["result"])
    else:
        None

# Call the app function to execute it
if __name__ == '__main__':
    app()
