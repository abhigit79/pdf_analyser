import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import time
import os
import io
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)
# Set the session timeout (in seconds)
# SESSION_TIMEOUT = 30 * 60  # 30 minutes timeout

# Track session start time
# if 'start_time' not in st.session_state:
#   st.session_state['start_time'] = time.time()


# Check for session timeout
# def is_session_expired():
#   elapsed_time = time.time() - st.session_state['start_time']


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Define a function to save files per user session
def save_uploaded_file(pdf_docs):
    # Create a unique session id for the user
    session_id = str(uuid.uuid4())  # Create a unique ID per session
    user_dir = f"./tmp/uploads/{session_id}"

    # Ensure the directory exists
    if not os.path.exists(user_dir):
        try:
            os.makedirs(user_dir)  # Try creating the directory
            # Optional: Make sure the directory is writable
            os.chmod(user_dir, 0o777)  # Grant read, write, and execute permissions
        except PermissionError as e:
            print(f"Error: Unable to create directory {user_dir}. {e}")
            return None, None

    # Save the uploaded file in the user's unique folder
    # file_path = os.path.join(user_dir, pdf_docs.name)

    # Save the file to disk
    # with open(user_dir, "wb") as f:
    # f.write(pdf_docs.getbuffer())

    return user_dir, session_id


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))  # Read the PDF file content
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Check if text extraction was successful
                    text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


def get_txt_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, user_dir):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(user_dir)


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say, "Answer is not available in the context."

    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, session_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(f"./tmp/uploads/{session_id}"):
        new_db = FAISS.load_local(f"./tmp/uploads/{session_id}", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply:", response["output_text"])
    else:
        st.error(f"Vector store not found. Please upload PDF files first.{session_id}")


def main():
    st.set_page_config("Chat with multiple PDFs")
    st.header("Chat with multiple PDFs using Gemini")
    session_id = None
    user_question = st.text_input("Ask a question from the PDF files")
    # Check if session_id exists in session_state (indicating that the user has uploaded files)
    session_id = st.session_state.get('session_id', None)
    if user_question:
        user_input(user_question, session_id)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the Submit", accept_multiple_files=True)
        if st.button("Submit"):
            if pdf_docs:  # Ensure files are uploaded
                user_dir, session_id = save_uploaded_file(pdf_docs)
                # Store session_id in session_state
                st.session_state['session_id'] = session_id
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_txt_chunk(raw_text)
                    get_vector_store(text_chunks, user_dir)
                    st.success("Done")

            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
