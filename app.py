import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import os
import numpy as np
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(" Gemini API key not found. Check your .env file.")
    st.stop()
# Load Gemini model
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("AI Document Q&A Study Buddy (Gemini Version)")
st.write("API KEY LOADED:", api_key is not None)
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
response = model_gemini.generate_content(prompt="Hello, Gemini! This is a test to check if the API key is working.")

if response.text:
    st.write(response.text)
else:
    st.warning(" No response from Gemini. Try rephrasing your question.")
if uploaded_file is not None:

    # Step 1: Read PDF
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    st.success(" PDF Loaded Successfully!")

    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    st.write(f" Total Chunks: {len(chunks)}")

    # Step 3: Convert chunks to embeddings
    embeddings = model.encode(chunks)

    # Step 4: Store in FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Step 5: Ask question
    query = st.text_input("Ask a question from the document:")

    if query:
        query_embedding = model.encode([query])

        # Step 6: Search similar chunks
        k = 3
        distances, indices = index.search(np.array(query_embedding), k)

        st.subheader(" Relevant Context:")
        context = ""

        for i in indices[0]:
            st.write(chunks[i])
            context += chunks[i] + "\n"

        #  Step 7: Gemini AI Answer
        st.subheader("AI Answer:")

        try:
            prompt = f"""
            Answer the question ONLY using the context below.
            If the answer is not present, say: "Answer not found in document."

            Context:
            {context}

            Question:
            {query}
            """

            response = model_gemini.generate_content(prompt)
            answer = response.text

            st.write(answer)

        except Exception as e:
            st.error(f" Error: {e}")
