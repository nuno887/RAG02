import os
import json
import sys
import argparse
from pathlib import Path
from pprint import pprint

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama model configuration
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
OLLAMA_URL = os.getenv('OLLAMA_URL')  # e.g. http://localhost:11434
# Embedding model configuration (must be an Ollama model that supports embeddings)
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mxbai-embed-large:latest')

# Default prompt template
default_template_str = '''
You are an excellent assistant. Your job is to search for information in the PDF documents, organize the information in the clearest form possible, and also provide the exact original text where relevant.
Metadata Header: {header}
Source file: {file}
Use the following context extracted from the documents to answer the question precisely.

Context:
{context}

Question: {question}
Answer:
'''
DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "header", "file"],
    template=default_template_str
)


def list_metadata_entries(json_dir: str) -> list[tuple[Path, dict]]:
    """
    Read metadata JSONs and return list of (segment_path, metadata) tuples.
    Converts list values to comma-separated strings.
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
                md['file'] = path.name
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(documents)

    # Use Ollama for embeddings, only pass base_url if defined
    if OLLAMA_URL:
        embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    else:
        embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

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
    # Use Ollama for embeddings, only pass base_url if defined
    if OLLAMA_URL:
        embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    else:
        embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embed_model,
        collection_name='rag_collection'
    )


def run_rag(query: str, vectorstore: Chroma, metadata_filter: dict = None, prompt_template: PromptTemplate = DEFAULT_PROMPT_TEMPLATE) -> str:
    """
    Execute RetrievalQA with Ollama using the configured model and prompt template.
    Optionally apply metadata filters.
    """
    # Instantiate LLM, passing base_url only if provided
    if OLLAMA_URL:
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    else:
        llm = Ollama(model=OLLAMA_MODEL)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 5}, filter=metadata_filter)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt_template}
    )
    result = qa.invoke({"query": query})
    return result.get("result", "")


def main():
    parser = argparse.ArgumentParser(description="Run the RAG system.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the vector store even if it exists")
    args = parser.parse_args()

    metadata_folder = os.getenv('JSON_EXPORTS', 'json_exports')
    db_folder = os.getenv('VECTOR_DB', 'chromadb')

    # List metadata segments
    print("=== Metadata Segments ===")
    entries = list_metadata_entries(metadata_folder)
    for idx, (path, _) in enumerate(entries, start=1):
        print(f"[{idx}] {path.name}")
    print(f"Total segments: {len(entries)}\n")

    # Build or load vector store
    if args.rebuild or not Path(db_folder).exists():
        action = "Rebuilding" if args.rebuild else "Building"
        print(f"{action} vector store from '{metadata_folder}' into '{db_folder}'...")
        db = build_vectorstore_from_metadata(metadata_folder, db_folder)
        print("Vector store built and persisted.\n")
    else:
        print(f"Loading existing vector store from '{db_folder}'...")
        db = load_vectorstore(db_folder)
        print("Vector store loaded.\n")

    # Interactive RAG loop
    print("=== RAG Query (type 'quit' to exit) ===")
    while True:
        question = input("Enter your question: ")
        if question.strip().lower() == 'quit':
            print("Exiting RAG system.")
            break
        header_key = input("Enter header key to filter by (or 'quit' to exit): ")
        if header_key.strip().lower() == 'quit':
            print("Exiting RAG system.")
            break
        filter_criteria = {'header': header_key}
        print(f"Running RAG query for header '{header_key}' and question '{question}'\n")
        answer = run_rag(question, db, metadata_filter=filter_criteria)

        # Display the answer
        print("\n=== RAG Answer ===")
        print(answer)
        print("\n=== Query Completed (type 'quit' to exit or new question) ===\n")

if __name__ == '__main__':
    main()
