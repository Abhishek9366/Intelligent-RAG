# Intelligent RAG: Zyro Dynamics Policy Assistant

A retrieval-augmented generation (RAG) assistant for answering employee questions from Zyro Dynamics internal policy documents. The project indexes the provided PDF corpus, retrieves the most relevant policy passages, and generates grounded responses through a Groq-hosted language model.

## Highlights

- Answers are grounded in the supplied policy PDFs rather than general model knowledge.
- Supports HR, compensation, travel, conduct, onboarding, and IT-security policy questions.
- Uses FAISS for vector similarity search and Hugging Face sentence-transformer embeddings.
- Includes scope handling for clearly unrelated questions.
- Produces a Kaggle-compatible `submission.csv` containing `question_id` and `answer`.

## Project structure

```text
.
|-- code.ipynb                       # End-to-end RAG workflow
|-- test.csv                         # Competition questions
|-- zyro-dynamics-hr-corpus/         # Source policy PDFs
|   |-- 00_Company_Profile.pdf
|   |-- 01_Employee_Handbook.pdf
|   `-- ...
|-- submission.csv                   
`-- .env                             
```

## How it works

```text
Policy PDFs -> PDF loader -> text chunks -> embeddings -> FAISS vector store
                                                            |
Employee question -> scope check -> relevant passages -> Groq LLM -> grounded answer
```

## Local setup

1. Create and activate a Python 3.11+ virtual environment.

2. Install the packages used by the notebook:

   ```bash
   pip install langchain langchain-community langchain-text-splitters \
     langchain-huggingface langchain-groq faiss-cpu pypdf \
     sentence-transformers python-dotenv
   ```

3. Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Open and run [code.ipynb](code.ipynb) from top to bottom.

The notebook loads the PDFs, creates the vector index, answers the questions in `test.csv`, and writes `submission.csv`.

## Tech stack

- Python
- LangChain
- FAISS
- Hugging Face sentence-transformers
- Groq API

