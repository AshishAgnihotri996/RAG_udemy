# RAG_udemy

A hands-on learning project for building a **Retrieval-Augmented Generation (RAG)** pipeline in Python with LangChain. The repository is at an early stage and currently covers the first step of a RAG system: **data ingestion and parsing**.

## What this project covers

- Understanding the LangChain `Document` structure (page content + metadata)
- Loading a single text file with `TextLoader`
- Loading multiple files from a folder with `DirectoryLoader`
- Setting up text splitters (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`) for chunking
- A project environment prepared for embeddings, vector stores and LLM providers

## Tech stack

| Area | Libraries |
|---|---|
| Framework | `langchain`, `langchain-community` |
| LLM providers | `langchain-groq`, `langchain-openai` |
| Vector stores | `chromadb`, `faiss-cpu` |
| Embeddings | `sentence-transformers` |
| Document handling | `pypdf`, `tiktoken`, `pandas` |
| Config and tooling | `python-dotenv`, `ipykernel`, `uv` |

## Project structure

```
RAG_udemy/
├── 0-DataParsing/
│   ├── 1-dataingestion.ipynb      # Data ingestion notebook
│   └── data/
│       └── textfiles/             # Sample .txt documents
│           ├── sample2.txt
│           ├── random_info_1_science_nature.txt
│           └── random_info_2_history_tech_world.txt
├── src/
│   └── rag_udemy/
│       └── __init__.py            # Package entry point (placeholder)
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Pip-compatible dependency list
├── uv.lock                        # Locked dependency versions
└── .python-version                # Python version (3.10)
```

## Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for any LLM provider you plan to use (for example Groq or OpenAI)

## Installation

**Option 1: using uv (recommended)**

```bash
git clone https://github.com/AshishAgnihotri996/RAG_udemy.git
cd RAG_udemy
uv sync
```

**Option 2: using pip**

```bash
git clone https://github.com/AshishAgnihotri996/RAG_udemy.git
cd RAG_udemy
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

If you use a hosted LLM, create a `.env` file in the project root and add your keys:

```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

The data ingestion notebook does not require API keys.

## Usage

1. Open the notebook:

   ```bash
   jupyter notebook 0-DataParsing/1-dataingestion.ipynb
   ```

2. Run the notebook from inside the `0-DataParsing` folder, since file paths like `data/textfiles` are relative.

3. The notebook walks through:
   - creating a `Document` with metadata
   - writing and loading a sample text file with `TextLoader`
   - loading every `.txt` file in a folder with `DirectoryLoader`

Example from the notebook:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "data/textfiles",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")
```

## Roadmap

- [ ] Chunk documents with text splitters
- [ ] Create embeddings with `sentence-transformers`
- [ ] Store and search vectors with FAISS or ChromaDB
- [ ] Build a retriever and connect it to an LLM (Groq / OpenAI)
- [ ] Add PDF loading with `pypdf`
- [ ] Evaluate retrieval quality

## Author

**Ashish Agnihotri**
GitHub: [@AshishAgnihotri996](https://github.com/AshishAgnihotri996)

## License

No license has been specified yet. Add a `LICENSE` file (for example MIT) to define how others can use this project.
