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
    text = []
    metadatas = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))  # Read the PDF file content
            # Check number of pages
            print(f"Number of pages in {pdf.name}: {len(pdf_reader.pages)}")
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:  # Check if text extraction was successful
                    text.append(page_text)
                    metadatas.append({"page_number": page_num + 1})  # Store page number in metadata
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    print(f"Extracted text: {len(text)} pages.")
    print(f"Extracted metadata: {len(metadatas)} entries.")
    return text, metadatas



def get_txt_chunk(text, metadatas):
    # Ensure the input is a list of text strings (one per page)
    if not all(isinstance(page_text, str) for page_text in text):
        raise ValueError("All elements of text must be strings.")
    if not all(isinstance(meta, dict) for meta in metadatas):
        raise ValueError("All elements of metadatas must be dictionaries.")

    # Combine the list of text strings into one large text block
    combined_text = "\n".join(text)
    text_chunks = []
    chunk_metadatas = []
    # Now use the text_splitter to split this large text into chunks

    for page_num, page_text in enumerate(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        text_chunks = text_splitter.split_text(page_text)

    # Now, we need to align chunks with metadatas (pages) for reference

        total_length = 0
        metadata_index = 0  # Start from the first page of metadata

        for chunk in text_chunks:
            page_number = metadatas[metadata_index]["page_number"]  # Default to the first page metadata
            chunk_start = total_length
            chunk_end = total_length + len(chunk)

        # Adjust which page this chunk belongs to by checking the range
            while metadata_index < len(metadatas) and chunk_end > total_length:
                page_text = text[metadata_index]  # Get the current page's text
                page_end = total_length + len(page_text)
                if chunk_end <= page_end:
                    break
                metadata_index += 1

                # Prevent out of range error by checking if we've gone beyond available metadata
                if metadata_index >= len(metadatas):
                    break

            # Ensure we append metadata only if the page number exists
            if metadata_index < len(metadatas):
                chunk_metadatas.append({"page_number": metadatas[metadata_index]["page_number"]})
            else:
                chunk_metadatas.append({"page_number": 1})  # Default to first page if out of bounds

            total_length += len(chunk)

    return text_chunks, chunk_metadatas


def get_vector_store(text_chunks, chunk_metadatas, user_dir):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Check if the lists are empty or have mismatched lengths
    print(f"Text chunks: {len(text_chunks)}")
    print(f"Chunk metadata: {len(chunk_metadatas)}")

    if len(text_chunks) != len(chunk_metadatas):
        raise ValueError("Number of text chunks does not match number of metadata entries.")

    # Ensure both are not empty before proceeding
    if not text_chunks:
        raise ValueError("Text chunks or metadata are empty!")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=chunk_metadatas)
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
        docs = new_db.similarity_search(user_question, k=1)

        # Extract the page number from the metadata of the document
        page_number = docs[0].metadata['page_number'] if 'page_number' in docs[0].metadata else "Unknown"


        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply:", response["output_text"])
        st.write("Refer to page:", page_number)
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
                    raw_text,metadatas = get_pdf_text(pdf_docs)
                    text_chunks,chunk_metadatas = get_txt_chunk(raw_text, metadatas)
                    get_vector_store(text_chunks, chunk_metadatas, user_dir)
                    st.success("Done")

            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
