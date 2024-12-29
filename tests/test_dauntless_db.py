import sys
sys.path.insert(0, '../')
from dauntless_db import DauntlessDB
import unittest
class TestDauntlessDB(unittest.TestCase):
    def setUp(self):
        """Set up a new instance of DauntlessDB before each test."""
        self.db = DauntlessDB()
        # Build vocabulary with some initial sentences for embedding generation
        initial_sentences_for_vocab_building = [
            "This is a sample sentence.",
            "This is another example.",
            "The cat sat on the mat.",
            "Dogs are great pets.",
            "Cats and dogs are friends."
        ]
        self.db.word2vec_model.build_vocab(initial_sentences_for_vocab_building)
    def tearDown(self):
        """Close the database connection after each test."""
        self.db.shutdown()
    def test_sql_operations(self):
        """Test basic SQL operations."""
        self.db.execute_sql("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
        self.db.execute_sql("INSERT INTO users (name) VALUES (?)", ("Alice",))

        # Fetch users
        users = self.db.execute_sql("SELECT * FROM users")
        self.assertEqual(users, [(1, 'Alice')], "User insertion failed!")
    def test_document_insert_and_retrieve(self):
        """Test inserting and retrieving documents."""
        self.db.insert_document("profiles", "user1", {"name": "Alice", "age": 30})

        document = self.db.get_document("profiles", "user1")
        self.assertEqual(document, {"name": "Alice", "age": 30}, "Document retrieval failed!")
    def test_document_delete(self):
        """Test deleting a document."""
        self.db.insert_document("profiles", "user1", {"name": "Alice", "age": 30})

        self.db.delete_document("profiles", "user1")
        deleted_document = self.db.get_document("profiles", "user1")

        self.assertIsNone(deleted_document, "Document deletion failed!")
    def test_sentence_embedding_insertion(self):
        """Test inserting sentences and generating embeddings."""
        self.db.insert_sentence("sentence1", "This is a sample sentence.")

        # Check if the embedding is generated and stored
        embedding = self.db.vector_store.get("sentence1")
        self.assertIsNotNone(embedding, "Sentence embedding was not generated.")
    def test_search_sentence(self):
        """Test searching for sentences based on embeddings."""
        self.db.insert_sentence("sentence1", "This is a sample sentence.")
        self.db.insert_sentence("sentence2", "This is another example.")
        search_results = self.db.search_sentence("sample")

        # Assert that we get results back
        self.assertGreater(len(search_results), 0, "Search returned no results!")

    def test_shutdown(self):
        """Test that the shutdown method closes the database connection properly."""
        try:
            self.db.shutdown()
            # If no exception occurs, the shutdown was successful
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Shutdown raised an exception: {e}")
if __name__ == "__main__":
    unittest.main()
