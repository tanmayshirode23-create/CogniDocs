# CogniDocs — AI-Powered Document Intelligence

**CogniDocs** is an AI-powered PDF question-answering application built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF documents, ask questions in natural language, and receive context-aware answers supported by relevant source passages.

The application combines semantic search with Google Gemini to retrieve document-specific information and generate useful answers.

## Features

- **PDF Upload:** Upload PDF documents through an interactive web interface.
- **Text Extraction:** Extract text and page metadata from PDF files.
- **Intelligent Chunking:** Divide documents into smaller, overlapping text chunks.
- **Semantic Search:** Find relevant document passages using vector embeddings and FAISS.
- **AI-Powered Answers:** Generate context-aware responses using Google Gemini.
- **Source References:** Display retrieved text passages and page information alongside answers.
- **Conversation Memory:** Maintain chat history to support follow-up questions.
- **Model Selection:** Choose from the supported Gemini chat models.
- **Interactive Interface:** Use a simple Streamlit interface to upload documents and interact with the AI assistant.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface |
| LangChain | Document processing and LLM integration |
| PyPDFLoader | PDF text extraction |
| RecursiveCharacterTextSplitter | Text chunking |
| Google Gemini Embeddings | Text vectorization |
| FAISS | Vector similarity search |
| Google Gemini | Answer generation |

## System Architecture

```text
                  USER
                   |
                   v
              Streamlit UI
                app.py
                   |
          +--------+---------+
          |                  |
          v                  v
     PDF Upload         User Question
          |                  |
          v                  v
    Text Extraction     Question Embedding
          |
          v
      Text Chunking
          |
          v
    Gemini Embeddings
          |
          v
      FAISS Index <------ Similarity Search
          |                       ^
          +-----------+-----------+
                      |
                      v
              Relevant Chunks
                      |
                      v
          Question + Context + History
                      |
                      v
               Google Gemini
                      |
                      v
            Answer + Source Passages
                      |
                      v
                Streamlit UI
```

## How It Works

CogniDocs follows a two-stage workflow: document indexing and question answering.

### 1. Document Indexing

**Step 1 — PDF Upload**

The user uploads a PDF through the Streamlit interface.

**Step 2 — Text Extraction**

PyPDFLoader extracts text from the PDF and preserves page metadata.

**Step 3 — Text Chunking**

RecursiveCharacterTextSplitter divides the extracted text into smaller, overlapping chunks. This makes document retrieval more focused while helping preserve context.

**Step 4 — Embedding Generation**

The Gemini embedding model converts each text chunk into a numerical vector representing its semantic meaning.

**Step 5 — Vector Indexing**

FAISS stores the chunk embeddings in a searchable vector index for efficient similarity retrieval.

### 2. Question Answering

**Step 1 — User Query**

The user enters a question about the uploaded document.

**Step 2 — Semantic Retrieval**

The question is converted into an embedding, and FAISS retrieves the most relevant document chunks.

**Step 3 — Context Construction**

The retrieved passages, user question, and conversation history are assembled for the language model.

**Step 4 — Answer Generation**

Google Gemini generates a natural-language answer using the provided context.

**Step 5 — Source Display**

The application displays the generated answer along with retrieved source passages and available page information, allowing the user to verify the response.

## Example Use Case

Imagine uploading an employee handbook and asking:

**Question:** What is the leave policy for employees?

CogniDocs processes the question by retrieving relevant passages from the handbook and supplying them to Gemini. The model generates an answer based on the retrieved information, and the application displays the response with supporting source passages.

The answer depends on the actual contents of the uploaded document.

## Project Structure

```text
CogniDocs/
├── app.py                 # Streamlit interface and application flow
├── rag_pipeline.py        # RAG pipeline and conversation memory
├── requirements.txt       # Python dependencies
├── .env.example           # Environment configuration template
├── .gitignore             # Git exclusions
├── company_handbook.pdf   # Sample PDF for testing
└── README.md              # Project documentation
```

## Installation and Setup

### Prerequisites

- Python 3.12
- Google Gemini API key
- Git (optional)
- Internet connection

### 1. Clone or Download the Project

```bash
git clone <YOUR_REPOSITORY_URL>
cd CogniDocs
```

Alternatively, download the project ZIP and extract it locally.

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API Key

Generate an API key through [Google AI Studio](https://aistudio.google.com/apikey).

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Ensure the application loads the environment variables and that `.env` is excluded from version control. Never commit or publicly share your API key.

### 5. Run the Application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

Upload a PDF, wait for indexing to finish, and start asking questions.

## Key Concepts

**Retrieval-Augmented Generation (RAG):** Combines document retrieval with language-model generation to provide answers grounded in external information.

**Embeddings:** Numerical representations of text used to identify semantically similar passages.

**FAISS:** A vector search library used to retrieve document chunks similar to a user's question.

**Chunking:** Splits long documents into smaller passages so relevant information can be retrieved efficiently.

**Conversation Memory:** Retains earlier messages to help the model interpret follow-up questions.

**Grounded Responses:** Uses retrieved document passages as context and displays sources to help users verify answers. This reduces the risk of unsupported responses but does not eliminate hallucinations.

## Limitations

- PDF extraction may be less accurate for scanned documents, complex layouts, or tables.
- Retrieval quality depends on document structure, chunking, and semantic similarity.
- Generated answers may contain unsupported information and should be checked against source passages.
- The application requires internet access and a working Gemini API key.
- The vector index is held in memory and is not persistent across application sessions.
- Document content and questions are sent to Google's API for processing.

## Future Enhancements

- Support multiple PDF documents in a single knowledge base.
- Persist vector indexes to avoid repeated embedding generation.
- Add automated tests for retrieval and answer quality.
- Improve handling of questions that cannot be answered from the document.
- Add streaming responses for a more interactive experience.
- Support OCR for scanned PDF documents.
- Improve source navigation and passage highlighting.

## Conclusion

CogniDocs demonstrates how Retrieval-Augmented Generation can make document information easier to access through natural-language questions. By combining Python, Streamlit, LangChain, FAISS, and Google Gemini, it provides a practical example of integrating document processing, semantic retrieval, and generative AI into one application.
