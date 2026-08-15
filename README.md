## Medical-Rag-Assistant 
An intelligent document-based question-answering system that uses Retrieval-Augmented Generation (RAG) to provide answers from medical documents with relevant source references.

## Live Demo
🔗 (https://pavithra-git162-medical-rag-assistant-app-nnnrzs.streamlit.app/)

## Features
-  Document-based question answering from medical documents.
-  AI chatbot for context-aware answers.
-  Semantic search using FAISS and Sentence Transformers.
-  Source references with document page numbers.
-  Deployed and accessible online.

## RAG System
This project uses a Retrieval-Augmented Generation pipeline combining:
### Document Retrieval
- Relevant document chunks are retrieved using semantic similarity search from the FAISS vector database.

### Embeddings
- The documents and user queries are converted into vector embeddings using Sentence Transformers.

### FAISS Vector Database
- FAISS is used to efficiently search for the most relevant document chunks based on semantic similarity.

### AI Response Generation
- The retrieved document content is passed to the Groq LLM as context to generate a natural-language answer.

### Source References
- The application displays the relevant document pages used to generate the answer, allowing users to trace the response back to the original documents.


  







  
