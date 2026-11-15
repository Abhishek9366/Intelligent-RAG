import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CORPUS_DIR = Path(__file__).resolve().parent / "zyro-dynamics-hr-corpus"

st.set_page_config(page_title="Zyro Dynamics HR Assistant", page_icon="🏢")
st.title("Zyro Dynamics HR Assistant 🏢")
st.markdown(
    "Ask me anything about Zyro Dynamics HR policies, travel, benefits, and conduct."
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
    st.stop()


@st.cache_resource(show_spinner="Loading HR policies and building the search index...")
def setup_rag_pipeline(api_key: str):
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"Policy corpus directory not found: {CORPUS_DIR}")

    documents = PyPDFDirectoryLoader(str(CORPUS_DIR)).load()
    if not documents:
        raise ValueError(f"No PDF documents found in {CORPUS_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=500)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The policy PDFs did not contain any readable text.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1024,
        groq_api_key=api_key,
    )
    return retriever, llm


try:
    retriever, model = setup_rag_pipeline(GROQ_API_KEY)
except Exception as exc:
    st.error(f"Unable to load the policy assistant: {exc}")
    st.stop()


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are the Zyro Dynamics internal policy assistant.

Answer only from the retrieved context. The context may include HR, IT security,
travel, onboarding, conduct, compensation, benefits, and company-profile documents.

Before answering, carefully check all retrieved passages for exact numbers, grade
rules, limits, eligibility requirements, dates, and timelines. Give the direct
answer first, then include only essential conditions.

Never invent information. If the answer is not explicitly supported by the context,
respond exactly:
"I couldn't find this information in the available internal policy documents."

Context:
{context}

Question: {question}

Answer:"""
)

GUARDRAIL_PROMPT = ChatPromptTemplate.from_template(
    """You classify questions for Zyro Dynamics' internal policy-document assistant.

Return IN_SCOPE for questions about Zyro Dynamics' HR, benefits, leave, payroll,
compensation, travel, expenses, workplace conduct, onboarding, resignation,
IT security, passwords, company systems, or facts that may be in the provided
Company Profile.

Return OUT_OF_SCOPE for questions about another company's policies, comparisons
with competitors or other products, general knowledge, coding, entertainment,
or personal/private employee data.

Reply with exactly: IN_SCOPE or OUT_OF_SCOPE

Question: {question}"""
)

REFUSAL_MESSAGE = (
    "I'm the Zyro Dynamics HR assistant and can help only with questions about "
    "company HR policies and procedures, such as leave, attendance, benefits, "
    "reimbursements, workplace conduct, and onboarding. Please ask an HR-related question."
)
NOT_FOUND_MESSAGE = (
    "I couldn't find this information in the available internal policy documents."
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_bot(question: str):
    guardrail_chain = GUARDRAIL_PROMPT | model | StrOutputParser()
    verdict = guardrail_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    docs = retriever.invoke(question)
    if not docs:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    chain = RAG_PROMPT | model | StrOutputParser()
    answer = chain.invoke(
        {"context": format_docs(docs), "question": question}
    ).strip()
    return {"answer": answer, "sources": docs}


def render_sources(docs):
    with st.expander("View Source Citations"):
        for index, doc in enumerate(docs, start=1):
            source_name = Path(doc.metadata.get("source", "Unknown Document")).name
            page_num = doc.metadata.get("page", "Unknown Page")
            # LangChain's PDF loaders store page numbers as zero-based indexes.
            if isinstance(page_num, int):
                page_num += 1
            st.markdown(f"**Source {index}: {source_name} (Page {page_num})**")
            st.caption(doc.page_content)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if prompt := st.chat_input("Ask about Zyro Dynamics policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                response = ask_bot(prompt)
                answer_text = response["answer"]
                sources = response["sources"]
            except Exception as exc:
                answer_text = f"I couldn't complete that request: {exc}"
                sources = []

        st.markdown(answer_text)
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
