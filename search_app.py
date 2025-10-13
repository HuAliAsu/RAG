import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

# Load environment variables (good practice)
load_dotenv()

# --- 1. CONFIGURATION ---
# IMPORTANT: These must match the settings used in your indexing script!
DB_PATH = "db_chroma_ollama_embeddinggemma"
EMBEDDING_MODEL_NAME = "embeddinggemma"

def main():
    """
    Main function to perform a semantic search on the existing vector database.
    """
    print("--- Starting Semantic Search Application ---")

    # --- 2. INITIALIZE EMBEDDING MODEL ---
    # We need the same embedding model to convert our query into a vector.
    print(f"Initializing Ollama embedding model: '{EMBEDDING_MODEL_NAME}'...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    print("Ollama embedding model initialized.")

    # --- 3. LOAD THE EXISTING VECTOR DATABASE ---
    # This is the core of the search app. We load the persisted database from disk.
    # Notice we are no longer using `from_documents`, as we are not creating, but loading.
    print(f"Loading vector database from '{DB_PATH}'...")
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    print("Vector database loaded successfully.")

    # --- 4. DEFINE QUERY AND EXECUTE SEARCH ---
    # This is where you input your question.
    query = "آدرس خونت کجاست"
    print(f"\nPerforming semantic search for query: '{query}'")

    start_time = time.time()
    # Use the `similarity_search` method to find the most relevant chunks.
    # `k` determines the number of results to return.
    results = vector_db.similarity_search(query, k=3)
    end_time = time.time()

    print(f"Search completed in {end_time - start_time:.2f} seconds.")

    # --- 5. DISPLAY THE RESULTS ---
    if not results:
        print("No relevant documents found.")
    else:
        print(f"\nFound {len(results)} relevant chunks:\n")
        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Content: {doc.page_content}")
            # You can also inspect the metadata if needed:
            # print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print("-" * 20)

if __name__ == "__main__":
    main()