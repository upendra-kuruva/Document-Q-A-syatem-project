import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
# Check if API key exists
if not api_key:
    st.error(" OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
    st.stop()

# Create OpenAI client
client = OpenAI(api_key=api_key)
st.write("API KEY:", api_key)
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title(" AI Document Q&A with PDF Upload")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

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

    st.write(f"📊 Total Chunks: {len(chunks)}")

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

        st.subheader("📖 Relevant Context:")
        context = ""

        for i in indices[0]:
            st.write(chunks[i])
            context += chunks[i] + "\n"

        # Step 7: Generate AI Answer
        st.subheader(" AI Answer:")

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer only from the given context. If not found, say 'Answer not found in document.'"
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }
                ]
            )

            answer = response.choices[0].message.content
            st.write(answer)

        except Exception as e:
            st.error(f" Error: {e}")
