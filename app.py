import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import traceback
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter as cts
from langchain.vectorstores import FAISS
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_community.embeddings.octoai_embeddings import OctoAIEmbeddings

def get_octoai_embeddings(texts, api_key):
    embedding_model = OctoAIEmbeddings(octoai_api_token=api_key)
    return embedding_model

def get_vectorstore(text_chunks, api_key):
    embeddings = get_octoai_embeddings(text_chunks, api_key)
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_text_chunks(text):
    text_splitter = cts(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_file_text(file_docs):
    text = ''
    for file in file_docs:
        filename = file.name
        if filename.endswith('.pdf'):
            pdfreader = PdfReader(file)
            for page in pdfreader.pages:
                text += page.extract_text()
        elif filename.endswith('.docx'):
            doc = Document(file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + '\n'
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title='ROCKY BOT', page_icon=':books:')
    st.header('Chat with Multiple DOCUMENTs :book:')
    
    
    # Change 1: Initialization checks added
    # Initialize session state variables
    if 'qa_history' not in st.session_state:
        st.session_state['qa_history'] = []
    if 'vector_storage' not in st.session_state:
        st.session_state['vector_storage'] = None
    if 'loaded_document' not in st.session_state:
        st.session_state['loaded_document'] = False

    # Take input as question
    question = st.text_input('Ask me a question about the document')
    
    
    
     # Change 2: Simplified the check for vector_storage
    if question and st.session_state['vector_storage']:
        vectorstore = st.session_state['vector_storage']
        response = vectorstore.similarity_search(question, k=6)
        octoai_api_key = os.getenv("OCTOAI_API_KEY")
        llm = OctoAIEndpoint(octoai_api_token=octoai_api_key, max_tokens=4000, temperature=0.1)
        output = llm.invoke(f"TASK:\n ANSWER THE QUESTIONS ACCORDING TO THE PROVIDED DOCUMENTS. \n QUESTION: {question}\nDOCUMENTS:{[response_ for response_ in response]}")
        st.session_state['qa_history'].append((question, output))

    with st.sidebar:
        st.subheader('Your Documents')
        file_docs = st.file_uploader('Upload your files here', accept_multiple_files=True, type=['pdf', 'docx'])

        if st.button('SUBMIT'):
            if file_docs:
                with st.spinner('Processing.....'):
                    try:
                        raw_text = get_file_text(file_docs)
                        text_chunks = get_text_chunks(raw_text)
                        octoai_api_key = os.getenv("OCTOAI_API_KEY")
                        if not octoai_api_key:
                            st.error("OCTOAI_API_KEY environment variable is not set.")
                            return

                        vectorstore = get_vectorstore(text_chunks, octoai_api_key)
                        st.success("Documents processed successfully.")
                        st.session_state['loaded_document'] = True
                        st.session_state['vector_storage'] = vectorstore

                    except Exception as e:
                        traceback.print_exc()
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please upload at least one document.")

    # Custom CSS for the boxes
    custom_css = """
    <style>
    .qa-box {
        position: relative;
        border: 2px solid #007BFF; /* Blue border */
        border-radius: 10px;
        background-color: #FFFFFF; /* White background */
        margin: 15px 0;
        padding: 15px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); /* Light shadow for better visibility */
    }
    .question {
        font-weight: bold;
        color: #007BFF; /* Blue color */
    }
    .answer {
        color: #000000; /* Orange-red color */
    }
    </style>

    """

    # Write the custom CSS to the Streamlit app
    st.markdown(custom_css, unsafe_allow_html=True)

    # Display the Q&A history in boxes
    if st.session_state['qa_history']:
        for i, (q, a) in enumerate(st.session_state['qa_history']):
            st.markdown(f"""
            <div class="qa-box">
                <p class="question">Q{i+1}: {q}</p>
                <p class="answer">{a}</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
