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

def generate_minutes(text_data, user_question, system_message_content, vector_store):
    prompt = f"{system_message_content}\nUser Question: {user_question}\nPlease answer the user question: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .txt or .docx file to chat with the document.")

    upload_file = st.file_uploader("Choose a file", type=["txt", "docx"])

    #with st.sidebar:
    # Chat option
    user_question = st.text_input("Ask a question about the document:")

    # Enable button only if file is uploaded
    if upload_file is not None:
        submit_button = st.button('Submit')
    else:
        submit_button = None

    if submit_button:
        # Read the uploaded file
        file_content = upload_file.read()
        
        if upload_file.type == "text/plain":  # If file is .txt
            text_data = file_content.decode("utf-8")
        elif upload_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # If file is .docx
            docx_data = BytesIO(file_content)
            text_data = docx2txt.process(docx_data)
        else:
            st.error("Unsupported file type. Please upload a .txt or .docx file.")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )

        texts = text_splitter.split_text(text_data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vector_store = FAISS.from_texts(texts, embeddings)

        # Access system message content
        system_message_content = "You are an AI assistant tasked with providing response from the provided text,  Please avoid unnecessary details or tangential points." 

        result = generate_response(text_data, user_question, system_message_content, vector_store)
        st.write("**Output:**")
        st.write(result["result"])
    # else:
    #     st.warning("Please upload a document to proceed.")

# Call the app function to execute it
if __name__ == '__main__':
    app()
