prokect tittle: RAG Assistant using Gemini, LangChain, FAISS, and Streamlit

Project Overview

This project is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload
PDF documents and ask questions about their content.
<img width="1357" height="780" alt="q a screen shots" src="https://github.com/user-attachments/assets/451a3055-522c-44b0-867a-2118446f5c61" />

The system extracts text from uploaded PDFs, splits the content into chunks, generates embeddings using Google's Gemini Embedding Model, 
stores them in a FAISS vector database, retrieves the most relevant chunks for a user query, and generates context-aware answers using Gemini 2.5 Flash.

Additionally, the project includes an automated AI Evaluation Module that measures the quality of generated responses using Groundedness, 
Context Relevance, and Answer Relevance metrics.

****Key Features

 ***PDF Document Processing
Upload PDF documents through Streamlit UI.
Extract text using PyPDFLoader.
Automatically parse and process document content.
*** Intelligent Text Chunking
Uses RecursiveCharacterTextSplitter.
Chunk Size: 1000 characters.
Chunk Overlap: 200 characters.
Preserves context across chunks.
*** Semantic Search
Generates vector embeddings using Gemini Embedding Model.
Stores embeddings in FAISS Vector Database.
Retrieves top-k relevant chunks for user queries.
***Context-Aware Question Answering
Uses Gemini 2.5 Flash LLM.
Answers are generated strictly from retrieved context.
Prevents hallucinations by restricting responses to document content.

 ****AI-Based Evaluation Framework
***Evaluates generated responses on:

**Groundedness
Measures whether the answer is supported by retrieved context.
**Context Relevance
Measures whether retrieved chunks are relevant to the query.
**Answer Relevance
Measures whether the generated answer addresses the user's question.

Skills Demonstrated

- Retrieval Augmented Generation (RAG)
- LangChain
- FAISS Vector Database
- Gemini API Integration
- Prompt Engineering
- Semantic Search
- Streamlit Application Development
- AI Evaluation Framework
- Vector Embeddings
- Document Question Answering (DocQA)

***System Architecture

User Uploads PDF
↓

PDF Loader (PyPDFLoader)
↓
Text Chunking
(RecursiveCharacterTextSplitter)
↓
Gemini Embeddings
↓
FAISS Vector Database
↓
User Query
↓
Similarity Search
↓
Relevant Chunks Retrieved
↓
Gemini 2.5 Flash
↓
Final Answer
↓
AI Evaluation Module


Project Workflow

Step 1: Upload PDF
The user uploads a PDF document through the Streamlit interface.

Step 2: Extract Document Content
PyPDFLoader extracts text from all pages of the document.

Step 3: Split Text into Chunks
The extracted text is divided into smaller chunks for efficient retrieval.

Step 4: Generate Embeddings
Gemini Embedding Model converts each text chunk into vector representations.

Step 5: Store in FAISS
All embeddings are stored in a FAISS vector database for fast similarity search.

Step 6: User Query
The user asks a question about the uploaded document.

Step 7: Retrieve Relevant Context
The retriever fetches the top 4 most relevant chunks from FAISS.

Step 8: Generate Answer
Gemini 2.5 Flash receives:

User Question

Retrieved Context
and generates a context-aware response.

Step 9: Evaluate Answer
The evaluation module assesses:

Groundedness

Context Relevance

Answer Relevance

and provides scoring with justifications.

****Technologies Used

Frontend

<img width="1482" height="776" alt="project imj 3" src="https://github.com/user-attachments/assets/2d54694a-e28c-43d3-a99a-913d70b5cad5" />
<img width="1395" height="606" alt="PROJECT imj 2" src="https://github.com/user-attachments/assets/4dead84c-189e-4857-a0a8-bebd4279d0fc" />
Streamlit

LLM

Gemini 2.5 Flash

Embedding Model

Gemini Embedding 2 Preview

Vector Database

FAISS

Frameworks

LangChain

PDF Processing

PyPDFLoader

Environment Management

Python Dotenv


****Sample Use Case

Upload:

Research Paper

Company Policy Document

Technical Documentation

User Manual

Ask:

"What are the main findings?"

"Summarize chapter 3."

"What does the policy say about leave management?"

"Explain the architecture discussed in the document."

The system retrieves relevant content and generates accurate answers grounded in the document.

****Future Enhancements

Multi-PDF Support

Chat History Memory

Hybrid Search (Keyword + Semantic)

Metadata Filtering

Citation-Based Answers

Persistent Vector Storage

Authentication and User Management

Evaluation Dashboard

Cloud Deployment (AWS/GCP/Azure)


***Author

Upendra Kuruva

AI/ML Enthusiast 
 
 Python Developer 
 
Generative AI Practitioner

Focused on building real-world AI applications using:

Python

Machine Learning

Generative AI

LangChain

Vector Databases

LLM Applications


