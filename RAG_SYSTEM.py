import os
import json
import re
import sys
from pathlib import Path
from pprint import pprint

# Ollama model configuration
MODEL_NAME = os.getenv('OLLAMA_MODEL', 'deepseek-r1:8b')

# Optional dotenv support
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# Ensure required libraries are installed
try:
    from langchain.schema import Document
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import Ollama
    from langchain.chains import RetrievalQA
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: pip install -r requirements.txt langchain-community")
    sys.exit(1)


def list_metadata_entries(json_dir: str) -> list[tuple[Path, dict]]:
    """
    Read metadata JSONs and return list of (segment_path, metadata) tuples.
    Converts list values to comma-separated strings for compatibility.
    """
    entries = []
    for jf in Path(json_dir).glob('*.json'):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        for header, records in data.items():
            for record in records:
                p = record.get('path')
                if not p:
                    continue
                path = Path(p)
                if not path.is_file():
                    continue
                md = {k: v for k, v in record.items() if k != 'path'}
                md['header'] = header
                # Convert lists to comma-separated strings
                for k, v in list(md.items()):
                    if isinstance(v, list):
                        md[k] = ", ".join(map(str, v))
                entries.append((path, md))
    return entries


def build_vectorstore_from_metadata(json_dir: str, persist_directory: str = 'chromadb') -> Chroma:
    """
    Build and persist a Chroma vector store from text segments listed in metadata.
    """
    entries = list_metadata_entries(json_dir)
    documents = []
    for path, md in entries:
        text = path.read_text(encoding='utf-8')
        documents.append(Document(page_content=text, metadata=md))

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embed_model,
        collection_name='rag_collection'
    )
    vectordb.add_documents(split_docs)
    vectordb.persist()
    return vectordb


def load_vectorstore(persist_directory: str = 'chromadb') -> Chroma:
    """
    Load an existing Chroma vector store.
    """
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embed_model,
        collection_name='rag_collection'
    )


def run_rag(query: str, vectorstore: Chroma, metadata_filter: dict = None) -> str:
    """
    Execute RetrievalQA with Ollama using the configured model.
    Optionally apply metadata filters.
    """
    ollama_url = os.getenv('OLLAMA_URL')
    if ollama_url:
        llm = Ollama(model=MODEL_NAME, base_url=ollama_url)
    else:
        llm = Ollama(model=MODEL_NAME)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 5}, filter=metadata_filter)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    return qa.run(query)


def main():
    load_dotenv()
    metadata_folder = os.getenv('JSON_EXPORTS', 'json_exports')
    db_folder = os.getenv('VECTOR_DB', 'chromadb')

    # List metadata entries
    print("=== Metadata Segments ===")
    entries = list_metadata_entries(metadata_folder)
    for idx, (path, md) in enumerate(entries, start=1):
        print(f"[{idx}] {path.name}")
    print(f"Total segments: {len(entries)}\n")

    # Build or load vector store
    if not Path(db_folder).exists():
        print(f"Building vector store from '{metadata_folder}' into '{db_folder}'...")
        db = build_vectorstore_from_metadata(metadata_folder, db_folder)
        print("Vector store built and persisted.\n")
    else:
        print(f"Loading existing vector store from '{db_folder}'...")
        db = load_vectorstore(db_folder)
        print("Vector store loaded.\n")

    # Prompt user for a RAG query and header key
    print("=== RAG Query ===")
    question = input("Enter your question: ")
    header_key = input("Enter header key to filter by (e.g. 'Despacho n.º 398/2025'): ")
    print(f"Running RAG query for header: {header_key} and question: '{question}'")
    filter_criteria = {'header': header_key}
    answer = run_rag(question, db, metadata_filter=filter_criteria)

    # Display the answer
    print("\n=== RAG Answer ===")
    print(answer)


if __name__ == '__main__':
    main()
