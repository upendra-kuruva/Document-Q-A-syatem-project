import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# Core LangChain components for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Modern Google GenAI SDK
from google import genai
from google.genai import types

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY. Please set it in your .env file.")
    st.stop()

# Initialize the official Gemini Client
client = genai.Client(api_key=api_key)


class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        
        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=text,
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=text,
        )
        return response.embeddings[0].values

def evaluate_response(query, context, answer):
    eval_prompt = f"""
    You are an expert AI Quality Assurance Judge. Rate the following RAG system response on a scale from 1 to 5 (with 5 being perfect) for three categories:
    
    1. Groundedness: Is the answer fully supported by the context? (1 = complete hallucination, 5 = 100% grounded)
    2. Context Relevance: Did the retrieved context contain the information needed to answer the query? (1 = irrelevant noise, 5 = highly relevant)
    3. Answer Relevance: Does the answer directly address the user's query? (1 = completely off-topic, 5 = perfectly answered)
    
    User Query: {query}
    Retrieved Context: {context}
    Generated Answer: {answer}
    
    Provide your response in clean Markdown with a brief 1-sentence justification for each score.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=eval_prompt,
    )
    return response.text
# --- Streamlit UI Configuration ---
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG System")
st.subheader("Upload a PDF and ask questions based on its content")

# Initialize session state for the vector database
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
    process_button = st.button("Process Document")

    if uploaded_file and process_button:
        with st.spinner("Processing PDF, creating chunks, and generating embeddings..."):
            try:
                # Save uploaded file temporarily to disk because PyPDFLoader requires a path
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Step 1: Load and parse PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                # Step 2: Split text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Step 3: Initialize Custom Gemini Embeddings and FAISS Vector DB
                embeddings = GeminiEmbeddings()
                vector_store = FAISS.from_documents(splits, embeddings)
                
                # Store the vector database in Streamlit's session state
                st.session_state.vector_store = vector_store
                st.success(f"Successfully processed {len(splits)} text chunks!")
                
                # Clean up the temporary file
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")


# --- Main Screen: Chat/QA System ---
if st.session_state.vector_store is not None:
    user_query = st.text_input("Ask a question about your document:")
    
    if user_query:
        # Initialize session state tracking keys for the current query
        if "current_query" not in st.session_state or st.session_state.current_query != user_query:
            st.session_state.current_query = user_query
            st.session_state.rag_answer = None
            st.session_state.rag_context = None
            st.session_state.eval_results = None  # Reset evaluation for a new question

        # Step 4 & 5 & 6: Only run the heavy RAG generation if we don't have an answer saved for this query
        if st.session_state.rag_answer is None:
            with st.spinner("Searching document and generating answer..."):
                # Retrieve relevant chunks
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(user_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Construct the prompt
                rag_prompt = f"""
                You are a helpful assistant. Answer the user's question accurately using ONLY the provided context below.
                If the answer cannot be found in the context, politely state that you do not know based on the provided document. Do not make up information.
                
                Context:
                {context}
                
                Question:
                {user_query}
                
                Answer:
                """
                
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=rag_prompt,
                    )
                    
                    # Lock the answers into Session State memory!
                    st.session_state.rag_answer = response.text
                    st.session_state.rag_context = context
                    st.session_state.relevant_docs = relevant_docs
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

        # --- Display Results from Session State (Safe from Reruns) ---
        if st.session_state.rag_answer:
            st.markdown("### Answer:")
            st.write(st.session_state.rag_answer)
            
            # Show references
            with st.expander("View Source Context Chunks"):
                for i, doc in enumerate(st.session_state.relevant_docs):
                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'unknown')}):**")
                    st.caption(doc.page_content)
                    st.markdown("---")

            # --- Evaluation Expander Block ---
            with st.expander("📊 Run AI Evaluation"):
                st.write("Click below to have Gemini review the performance metrics for this retrieval turnaround.")
                
                # Check if we already have eval results saved to avoid double API calls on eval itself
                if st.session_state.eval_results:
                    st.markdown("---")
                    st.markdown(st.session_state.eval_results)
                else:
                    if st.button("Run Evaluation Metrics"):
                        with st.spinner("Analyzing accuracy matrix..."):
                            # Pass the frozen variables safely from session state
                            eval_results = evaluate_response(
                                st.session_state.current_query, 
                                st.session_state.rag_context, 
                                st.session_state.rag_answer
                            )
                            st.session_state.eval_results = eval_results
                            st.rerun()  # Forces a quick clean UI update to show the evaluation results
                            
else:
    st.info("Please upload and process a PDF document via the sidebar to start querying.")
