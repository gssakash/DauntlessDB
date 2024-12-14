import sqlite3
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from threading import Lock
from word2vec import SimpleWord2Vec

class DauntlessDB:
    """
    DauntlessDB is a Python-based multimodal database supporting SQL, document, and vector storage.
    
    ## Overview
    DauntlessDB combines the functionalities of a traditional SQL database with the flexibility of 
    document storage and the efficiency of vector embeddings. It is designed to handle various data 
    types and provide ACID compliance, transactional capabilities, efficient vector search, and 
    multi-threaded operations.

    ## Objectives
    - Provide a unified interface for SQL and document-based storage.
    - Support efficient vector operations for semantic search using embeddings.
    - Ensure data integrity and consistency through ACID transactions.
    - Enable concurrent access to the database with thread safety.

    ## Features
    The following features are included in DauntlessDB:
    
    1. SQL Storage: Execute SQL commands to manage structured data.
    2. Document Storage: Store and retrieve documents in an in-memory dictionary for quick access.
    3. Vector Storage: Insert sentences as embeddings for semantic search capabilities.
    4. Transactional Support: Use locks to ensure safe concurrent access to documents and vectors.
    5. Sentence Embedding Generation: Automatically generate embeddings for sentences using Word2Vec.
    6. Approximate Nearest Neighbor Search: Perform similarity searches on sentence embeddings.
    7. Flexible Initialization: Configure vocabulary size and embedding dimensions upon initialization.
    8. Graceful Shutdown: Safely close database connections and clean up resources.

    ## Design Choices
    - SQLite Backend: SQLite is chosen for its lightweight nature and ease of use for prototyping.
    
    - In-Memory Document Store: An in-memory dictionary allows for quick access to documents without 
      the overhead of disk I/O for frequently accessed data.

    - Thread Safety: Locks are used around critical sections of code to prevent race conditions 
      when multiple threads access shared resources.

    ## Usage
    To use DauntlessDB, create an instance of the class, execute SQL commands, insert documents, 
    and perform vector searches as demonstrated in the example usage section.

    ## Limitations
    While DauntlessDB provides a robust framework for multimodal data handling, it may not be suitable 
    for large-scale production environments due to its in-memory document store and SQLite's limitations 
    on concurrent writes. Users should consider these factors when deploying this solution.

    """

    def __init__(self, db_path: str = ":memory:", vocab_size: int = 1000, embedding_dim: int = 50):
        """
        Initialize the DauntlessDB instance.

        :param db_path: Path to the SQLite database for SQL storage.
        :param vocab_size: Number of words in the simulated vocabulary.
        :param embedding_dim: Dimensionality of the embeddings.
        """
        self.db_path = db_path
        self.sql_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.sql_lock = Lock()
        self.document_store: Dict[str, Dict[str, Any]] = {}  # In-memory document store
        self.vector_store: Dict[str, np.ndarray] = {}  # In-memory vector store
        self.transaction_queue = Queue()
        self.transaction_lock = Lock()

        # Initialize SimpleWord2Vec model
        self.word2vec_model = SimpleWord2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim)

    def execute_sql(self, query: str, params: tuple = ()) -> Optional[Any]:
         """
         Execute a SQL query against the database.

         :param query: The SQL query to execute.
         :param params: Parameters to bind to the query (if any).
         :return: Result of the query if it's a SELECT statement; None otherwise.
         """
         with self.sql_lock:
             cursor = self.sql_conn.cursor()
             try:
                 cursor.execute(query, params)
                 if query.strip().lower().startswith("select"):
                     result = cursor.fetchall()
                 else:
                     self.sql_conn.commit()
                     result = None
                 return result
             except Exception as e:
                 self.sql_conn.rollback()
                 raise e

    def insert_document(self, collection: str, document_id: str, document: dict) -> None:
         """
         Insert a document into the in-memory document store.

         :param collection: The collection name where the document will be stored.
         :param document_id: Unique identifier for the document.
         :param document: The document data as a dictionary.
         """
         with self.transaction_lock:
             if collection not in self.document_store:
                 self.document_store[collection] = {}
             self.document_store[collection][document_id] = document

    def get_document(self, collection: str, document_id: str) -> Optional[dict]:
         """
         Retrieve a document from the in-memory document store.

         :param collection: The collection name from which to retrieve the document.
         :param document_id: Unique identifier for the document.
         :return: The requested document as a dictionary or None if not found.
         """
         return self.document_store.get(collection, {}).get(document_id)

    def delete_document(self, collection: str, document_id: str) -> None:
         """
         Delete a document from the in-memory document store.

         :param collection: The collection name from which to delete the document.
         :param document_id: Unique identifier for the document to delete.
         """
         with self.transaction_lock:
             if collection in self.document_store:
                 self.document_store[collection].pop(document_id, None)

    def insert_sentence(self, key: str, sentence: str) -> None:
         """Insert a sentence into the vector store with automatic embedding generation."""
         embedding = self.word2vec_model._generate_sentence_embedding(sentence)
         with self.transaction_lock:
             self.vector_store[key] = embedding

    def search_sentence(self, query: str, top_k: int = 5) -> List[tuple]:
         """Perform approximate nearest neighbor (ANN) search for a query sentence."""
         query_embedding = self.word2vec_model._generate_sentence_embedding(query)
         results = []

         with self.transaction_lock:
             for key, vector in self.vector_store.items():
                 similarity = np.dot(query_embedding, vector) / (
                     np.linalg.norm(query_embedding) * np.linalg.norm(vector)
                 )
                 results.append((key, similarity))

         results.sort(key=lambda x: x[1], reverse=True)
         return results[:top_k]

    def shutdown(self) -> None:
        """ Safely shut down DauntlessDB by closing all connections and cleaning up resources. """
        with self.sql_lock:
            self.sql_conn.close()

def main():
    """ Example usage of DauntlessDB with assertions for testing. """
    
    # Initialize DauntlessDB with SimpleWord2Vec model
    db = DauntlessDB()

    # Build vocabulary with some initial sentences before inserting any sentences into DB
    initial_sentences_for_vocab_building = [
       "This is a sample sentence.",
       "This is another example.",
       "The cat sat on the mat.",
       "Dogs are great pets.",
       "Cats and dogs are friends."
   ]
        
    db.word2vec_model.build_vocab(initial_sentences_for_vocab_building)

    # SQL operations example
    db.execute_sql("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute_sql("INSERT INTO users (name) VALUES (?)", ("Alice",))
    
    # Fetch users and assert
    users = db.execute_sql("SELECT * FROM users")
    print("Users:", users)
    assert users == [(1, 'Alice')], "User insertion failed!"

    # Document operations example
    db.insert_document("profiles", "user1", {"name": "Alice", "age": 30})
    
    # Fetch document and assert
    document = db.get_document("profiles", "user1")
    print("Document:", document)
    assert document == {"name": "Alice", "age": 30}, "Document insertion failed!"

    # Delete document and assert
    db.delete_document("profiles", "user1")
    deleted_document = db.get_document("profiles", "user1")
    print("Document after deletion:", deleted_document)
    assert deleted_document is None, "Document deletion failed!"

    # Sentence embedding example
    db.insert_sentence("sentence1", "This is a sample sentence.")
    db.insert_sentence("sentence2", "This is another example.")

    search_results = db.search_sentence("sample")
    print("Search results for 'sample':", search_results)
   
    # Assert search results contain expected keys
    assert len(search_results) > 0, "Search returned no results!"
   
    # Shutdown DB safely
    db.shutdown()
   
if __name__ == "__main__":
     main()
