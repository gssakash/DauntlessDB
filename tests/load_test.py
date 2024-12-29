import sys
sys.path.insert(0, '../')  # Adjust this path if necessary
from dauntless_db import DauntlessDB
import pytest
import time

@pytest.fixture
def dauntless_db():
    db = DauntlessDB()
    # Build vocabulary with some initial sentences for embedding generation
    initial_sentences_for_vocab_building = [
        "This is a sample sentence.",
        "This is another example.",
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "Cats and dogs are friends."
    ]
    db.word2vec_model.build_vocab(initial_sentences_for_vocab_building)
    yield db
    db.shutdown()

def test_sql_operations(dauntless_db):
    for _ in range(100):
        start_time = time.perf_counter()
        dauntless_db.execute_sql("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
        dauntless_db.execute_sql("INSERT INTO users (name) VALUES (?)", (f"User_{_}",))
        users = dauntless_db.execute_sql("SELECT * FROM users")
        end_time = time.perf_counter()
        print(f"SQL operation {_} took {end_time - start_time:.4f} seconds")
        assert len(users) == _ + 1, "User insertion failed!"

def test_document_insert_and_retrieve(dauntless_db):
    for _ in range(100):
        start_time = time.perf_counter()
        dauntless_db.insert_document("profiles", f"user_{_}", {"name": f"User_{_}", "age": 30})
        document = dauntless_db.get_document("profiles", f"user_{_}")
        end_time = time.perf_counter()
        print(f"Document insert and retrieve {_} took {end_time - start_time:.4f} seconds")
        assert document == {"name": f"User_{_}", "age": 30}, "Document retrieval failed!"

def test_document_delete(dauntless_db):
    for _ in range(100):
        start_time = time.perf_counter()
        dauntless_db.insert_document("profiles", f"user_{_}", {"name": f"User_{_}", "age": 30})
        dauntless_db.delete_document("profiles", f"user_{_}")
        deleted_document = dauntless_db.get_document("profiles", f"user_{_}")
        end_time = time.perf_counter()
        print(f"Document deletion {_} took {end_time - start_time:.4f} seconds")
        assert deleted_document is None, "Document deletion failed!"

def test_sentence_embedding_insertion(dauntless_db):
    for _ in range(100):
        start_time = time.perf_counter()
        dauntless_db.insert_sentence(f"sentence_{_}", f"This is a sample sentence {_}.")
        embedding = dauntless_db.vector_store.get(f"sentence_{_}")
        end_time = time.perf_counter()
        print(f"Sentence embedding insertion {_} took {end_time - start_time:.4f} seconds")
        assert embedding is not None, "Sentence embedding was not generated."

def test_search_sentence(dauntless_db):
    for _ in range(100):
        start_time = time.perf_counter()
        dauntless_db.insert_sentence(f"sentence_{_}", f"This is a sample sentence {_}.")
        dauntless_db.insert_sentence(f"sentence_{_+1}", f"This is another example {_}.")
        search_results = dauntless_db.search_sentence("sample")
        end_time = time.perf_counter()
        print(f"Search sentence {_} took {end_time - start_time:.4f} seconds")
        assert len(search_results) > 0, "Search returned no results!"

# Run tests in parallel using pytest-xdist
import time
import pytest

start_time = time.perf_counter()

if __name__ == "__main__":
    pytest.main(["-n", "auto", "-v", "tests/test_dauntless_db.py"])

end_time = time.perf_counter()
print(f"Total test suite execution time: {end_time - start_time:.4f} seconds")
