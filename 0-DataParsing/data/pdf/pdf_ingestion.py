"""
PDF ingestion and parsing with LangChain
----------------------------------------
Install:
    pip install langchain-community langchain-text-splitters pypdf pdfplumber

Optional (vector store step):
    pip install langchain-openai faiss-cpu
    # or free local embeddings:
    pip install langchain-huggingface sentence-transformers faiss-cpu

Usage:
    python pdf_ingestion.py                      # uses sample_rag_guide.pdf
    python pdf_ingestion.py path/to/your.pdf
"""

import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample_rag_guide.pdf"


def line(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------
# 1. PDF metadata (pypdf)
# ---------------------------------------------------------------
def show_metadata(path):
    line("1. PDF METADATA")
    reader = PdfReader(path)
    meta = reader.metadata
    print("Pages   :", len(reader.pages))
    print("Title   :", meta.title if meta else None)
    print("Author  :", meta.author if meta else None)
    print("Encrypted:", reader.is_encrypted)


# ---------------------------------------------------------------
# 2. Load PDF as LangChain Documents (one Document per page)
# ---------------------------------------------------------------
def load_pdf(path):
    line("2. LOAD PDF  (PyPDFLoader: 1 Document per page)")
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    print("Full metadata of page 1 (kept on every chunk):", docs[0].metadata)
    for d in docs:
        print(f"\n--- Page {d.metadata.get('page', 0) + 1} | source: {d.metadata.get('source')} ---")
        print(d.page_content)
    return docs


# ---------------------------------------------------------------
# 3. Clean the extracted text
# ---------------------------------------------------------------
def clean_docs(docs):
    line("3. CLEAN TEXT")
    for d in docs:
        text = d.page_content
        text = text.replace("\x00", "")                 # null bytes
        text = " ".join(text.split())                   # collapse whitespace/newlines
        d.page_content = text
    print("Whitespace normalised on", len(docs), "pages")
    return docs


# ---------------------------------------------------------------
# 4. Split into chunks (keeps page metadata)
# ---------------------------------------------------------------
def split_docs(docs, chunk_size=400, chunk_overlap=60):
    line(f"4. SPLIT INTO CHUNKS (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(c.page_content)} chars) ---")
        print("page:", c.metadata.get("page", 0) + 1, "| start_index:", c.metadata.get("start_index"))
        print(c.page_content)
    return chunks


# ---------------------------------------------------------------
# 5. Extract TABLES (pdfplumber) - PyPDFLoader flattens tables
# ---------------------------------------------------------------
def extract_tables(path):
    line("5. TABLE EXTRACTION (pdfplumber)")
    found = 0
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            for table in page.extract_tables():
                found += 1
                print(f"\nTable {found} on page {page_no}:")
                for row in table:
                    print(row)
    if not found:
        print("No tables found.")


# ---------------------------------------------------------------
# 6. Detect scanned PDFs (no text layer -> needs OCR)
# ---------------------------------------------------------------
def check_scanned(docs):
    line("6. SCANNED-PDF CHECK")
    empty = [d.metadata.get("page", 0) + 1 for d in docs if len(d.page_content.strip()) < 20]
    if empty:
        print(f"Pages with little/no text: {empty}")
        print("These are probably scans. Use OCR, e.g. pytesseract + pdf2image,")
        print("or UnstructuredPDFLoader(path, strategy='ocr_only').")
    else:
        print("All pages have a text layer. No OCR needed.")


# ---------------------------------------------------------------
# 7. Load a whole folder of PDFs
# ---------------------------------------------------------------
def load_folder(folder="."):
    line(f"7. LOAD ALL PDFs IN FOLDER: {Path(folder).resolve()}")
    loader = PyPDFDirectoryLoader(folder)
    docs = loader.load()
    sources = sorted({d.metadata["source"] for d in docs})
    print(f"{len(docs)} pages from {len(sources)} PDF file(s):")
    for s in sources:
        print(" -", s)
    return docs


# ---------------------------------------------------------------
# 8. Embed + store + search (optional; needs an embeddings model)
# ---------------------------------------------------------------
def build_vector_store(chunks):
    line("8. EMBED + VECTOR STORE + SEARCH")
    from langchain_community.vectorstores import FAISS

    # Option A: OpenAI (needs OPENAI_API_KEY)
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Option B: free, local
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local("faiss_index")
    print("Saved index to ./faiss_index")

    query = "Which chunking strategy is best for Markdown?"
    print(f"\nQuery: {query}")
    for r in store.similarity_search(query, k=2):
        print(f"\n[page {r.metadata.get('page', 0) + 1}] {r.page_content}")
    return store


if __name__ == "__main__":
    print("PDF file:", PDF_PATH)

    show_metadata(PDF_PATH)
    docs = load_pdf(PDF_PATH)
    check_scanned(docs)
    docs = clean_docs(docs)
    chunks = split_docs(docs)
    extract_tables(PDF_PATH)
    # load_folder(".")            # uncomment to ingest every PDF in a folder
    # build_vector_store(chunks)  # uncomment once embeddings are installed
