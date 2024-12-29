# DauntlessDB: A Lightweight Multimodal Database

## Overview
DauntlessDB is a lightweight multimodal database designed to provide a flexible and efficient solution for managing various data types, including structured, semi-structured, and unstructured data. It aims to mimic industry leaders like SurrealDB by integrating SQL, document storage, and vector embeddings into a single platform. This makes DauntlessDB suitable for light workloads and applications that require quick access to diverse data formats without the overhead of more complex systems.

## Objectives
- **Unified Data Management**: Provide a single interface for managing SQL, document, and vector data.
- **Lightweight Architecture**: Ensure minimal resource consumption while maintaining performance.
- **Flexible Querying**: Allow users to run queries across different data models seamlessly.
- **Transactional Integrity**: Support ACID transactions to ensure data consistency and reliability.

## Features
DauntlessDB includes a range of features designed to enhance usability and functionality:

1. **Multimodal Data Support**: Handle various data types (documents, vectors) in a unified environment without needing complex transformations.
2. **SQL Interface**: Execute standard SQL queries for structured data management.
3. **Document Storage**: Store documents in an in-memory dictionary for fast access and retrieval.
4. **Vector Embedding Generation**: Automatically generate embeddings for sentences using the integrated SimpleWord2Vec model.
5. **Approximate Nearest Neighbor Search**: Perform efficient similarity searches on vector embeddings.
6. **Thread Safety**: Utilize locks to manage concurrent access to shared resources, ensuring safe multi-threaded operations.
7. **Graceful Shutdown**: Safely close database connections and clean up resources when shutting down.

## Limitations
While DauntlessDB provides a robust framework for multimodal data handling, it has certain limitations:

1. **In-Memory Document Store**: The reliance on an in-memory store may limit scalability and persistence compared to disk-based solutions.
2. **Limited Concurrency Support**: Although thread-safe, the current implementation may not handle high levels of concurrent writes efficiently.
3. **Basic Querying Capabilities**: The querying capabilities may not be as advanced as those found in more established multi-model databases like SurrealDB or MarkLogic.
4. **Lack of Advanced Features**: Features such as full-text search indexing or complex transaction handling may be limited compared to industry leaders.

## Design Choices
DauntlessDB is built with several key design principles in mind:

- **Simplicity and Lightweight Design**: The architecture is designed to be lightweight, making it suitable for applications with moderate data needs without the complexity of larger systems.
- **Focus on Flexibility**: By supporting multiple data models within a single database, DauntlessDB allows users to adapt their data management strategies as requirements evolve.

# DauntlessDB Usage Documentation

## Overview
This documentation provides a comprehensive guide to using DauntlessDB, a versatile database system with advanced features like SQL operations, document management, and semantic search capabilities.

## Initialization
```python
 db = DauntlessDB()
```
Creates a new DauntlessDB instance with an integrated SimpleWord2Vec model.

## Vocabulary Building
### Initial Vocabulary Setup

```python
initial_sentences = [
"This is a sample sentence.",
"This is another example.",
"The cat sat on the mat.",
"Dogs are great pets.",
"Cats and dogs are friends."
]
db.word2vec_model.build_vocab(initial_sentences)
```
Prepares the vocabulary for semantic processing and search functionality.

## SQL Operations
### Creating Tables
```python
 db.execute_sql("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
```
### Inserting Records
```python
db.execute_sql("INSERT INTO users (name) VALUES (?)", ("Alice",))
```
### Fetching Records
```python
users = db.execute_sql("SELECT * FROM users")
```

## Document Management
### Inserting Documents
```python
db.insert_document("profiles", "user1", {"name": "Alice", "age": 30})
```

### Retrieving Documents
```python
document = db.get_document("profiles", "user1")
```

### Deleting Documents
```python
db.delete_document("profiles", "user1")
```

## Semantic Search
### Inserting Sentences
```python
db.insert_sentence("sentence1", "This is a sample sentence.")
db.insert_sentence("sentence2", "This is another example.")
```

### Searching Sentences
```python
search_results = db.search_sentence("sample")
```

## Database Shutdown
```python
db.shutdown()
```
Safely closes the database connection and releases resources.

## Best Practices
- Always build vocabulary before performing semantic searches
- Use transactions for complex SQL operations
- Handle potential exceptions during database interactions
- Call `shutdown()` to ensure proper resource management

## Conclusion
DauntlessDB serves as an innovative solution for users seeking a lightweight multimodal database that integrates the best features of SQL, document storage, and vector embeddings. While it is designed for light workloads and quick access to diverse data formats, users should be aware of its limitations regarding scalability and advanced querying capabilities. As the demand for flexible data solutions grows, DauntlessDB positions itself as a practical choice for developers looking to streamline their data management processes.
