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
        filename=file.name
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
    st.header('Chat with Multiple PDFs :book:')
    
    # ADD COLUMN THAT SAVES QUESTION AND ANSWERS HISTORY
    if 'qa_history' not in st.session_state:
        st.session_state['qa_history'] = []
    
    # TAKE INPUT AS QUESTION
    question = st.text_input('Ask me a question about the document')

    if 'vector_storage' in st.session_state:
        vectorstore = st.session_state['vector_storage']
    else:
        vectorstore = None

    if question and vectorstore:
        similar_docs = vectorstore.similarity_search(question)
        response = vectorstore.similarity_search(question, k=6)
        octoai_api_key = os.getenv("OCTOAI_API_KEY")
        llm = OctoAIEndpoint(octoai_api_token=octoai_api_key, max_tokens=4000, temperature=0.1)
        output = llm.invoke(f"TASK:\n ANSWER THE QUESTIONS ACCORDING TO THE PROVIDED DOCUMENTS. \n QUESTION: {question}\nDOCUMENTS:{[response_ for response_ in response]}")
        st.session_state['qa_history'].append((question, output))

    with st.sidebar:
        st.subheader('Your Documents')
        file_docs = st.file_uploader('Upload your PDF files here', accept_multiple_files=True, type=['pdf', 'docx'])

        if st.button('Process'):
            if file_docs:
                with st.spinner('Processing...'):
                    try:
                        raw_text = get_file_text(file_docs)
                        text_chunks = get_text_chunks(raw_text)
                        octoai_api_key = os.getenv("OCTOAI_API_KEY")
                        if not octoai_api_key:
                            st.error("OCTOAI_API_KEY environment variable is not set.")
                            return

                        vectorstore = get_vectorstore(text_chunks, octoai_api_key)
                        st.success("PDF processed successfully.")
                        st.session_state['loaded_document'] = True
                        st.session_state['vector_storage'] = vectorstore

                    except Exception as e:
                        traceback.print_exc()
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please upload at least one PDF file.")


# display all answers with questions
    if st.session_state['qa_history']:
        for i, (q, a) in enumerate(st.session_state['qa_history']):
            st.write(f"**Q{i+1}: {q}**")
            st.write(f"A{i+1}: {a}")

if __name__ == '__main__':
    main()
