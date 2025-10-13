import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

# --- 1. CONFIGURATION ---
DATA_PATH = "data"
# It's a good idea to name the DB after the model used to create it.
DB_PATH = "db_chroma_ollama_embeddinggemma"
# This is the model name as registered in Ollama (from `ollama list`)
EMBEDDING_MODEL_NAME = "embeddinggemma"

def main():
    """
    Main function to execute the full document indexing pipeline using Ollama.
    """
    print("--- Starting Document Indexing Pipeline with Ollama ---")
    start_time = time.time()

    # --- 2. SETUP & VALIDATION ---
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' is empty or does not exist.")
        return

    # --- 3. LOADING DOCUMENTS ---
    print(f"\n[Step 1/4] Loading documents from '{DATA_PATH}'...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print("No documents were loaded.")
        return
    print(f"Successfully loaded {len(documents)} document(s).")

    # --- 4. INITIALIZING OLLAMA EMBEDDING MODEL ---
    print(f"\n[Step 2/4] Initializing Ollama embedding model: '{EMBEDDING_MODEL_NAME}'...")
    # This interface connects to the Ollama service running in the background.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    print("Ollama embedding model initialized.")

    # --- 5. TEXT SPLITTING (SEMANTIC CHUNKING) ---
    print("\n[Step 3/4] Splitting documents using SemanticChunker (this may take a while)...")
    semantic_splitter = SemanticChunker(embeddings)
    chunks = semantic_splitter.split_documents(documents)
    print(f"Documents successfully split into {len(chunks)} semantic chunks.")

    # --- 6. CREATING AND PERSISTING THE VECTOR DATABASE ---
    print(f"\n[Step 4/4] Creating and persisting vector database to '{DB_PATH}'...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vector database created and persisted successfully.")

    # --- 7. COMPLETION ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n--- Document Indexing Pipeline Finished ---")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Vector database saved at: '{DB_PATH}'")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")


if __name__ == "__main__":
    main()