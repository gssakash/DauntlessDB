from typing import List, Tuple
import numpy as np

class SimpleWord2Vec:
    """
    A simplified implementation of the Word2Vec model for generating word embeddings.

    ## Overview
    The SimpleWord2Vec class is designed to provide a basic framework for learning word embeddings
    from a corpus of text. Word embeddings are dense vector representations of words that capture
    semantic relationships and contextual similarities. This implementation follows the skip-gram
    approach, which predicts context words given a target word.

    ## Objectives
    - Vocabulary Building: Efficiently build a vocabulary from input sentences, mapping words
      to unique indices.

    - Training Data Generation: Create training pairs of target words and their context words
      based on a specified window size.

    - Embedding Training: Train the model using gradient descent to adjust the embeddings based on
      the similarity between target and context words.

    - Vector Retrieval: Allow users to retrieve the vector representation of individual words.

    - Sentence Embedding: Provide functionality to generate a sentence-level embedding by averaging
      the embeddings of its constituent words.

    ## Design Choices

    - Vocabulary Size: The `vocab_size` parameter limits the number of unique words to consider,
      which helps manage memory and computational efficiency. This is particularly useful when working
      with large datasets where not all words may be relevant.

    - Embedding Dimension: The `embedding_dim` parameter allows customization of the size of the
      word vectors. A higher dimensionality can capture more complex relationships but may also lead
      to overfitting, especially with limited data.

    - Window Size: The `window_size` parameter defines the number of context words to consider
      around each target word. This choice can impact the quality of learned embeddings; a larger window
      captures broader context, while a smaller window focuses on immediate neighbors.

    ## Functions
    - `_tokenize(text: str) -> List[str]`: Tokenizes input text into lowercase words for uniformity.

    - `build_vocab(sentences: List[str]) -> None`: Constructs a vocabulary from the provided sentences,
      counting word occurrences and mapping them to indices.

    - `_generate_training_data(sentences: List[str]) -> List[Tuple[int, List[int]]]`: Creates pairs of
      target and context word indices based on the specified window size.

    - `train(sentences: List[str], epochs: int = 1000) -> None`: Trains the Word2Vec model using the
      generated training data over a specified number of epochs, adjusting embeddings through gradient descent.

    - `get_vector(word: str) -> np.ndarray`: Returns the vector representation for a given word.

    - `_generate_sentence_embedding(sentence: str) -> np.ndarray`: Computes an average embedding for an
      input sentence based on its constituent word vectors.

    ## Usage
    To use this class, instantiate it with desired parameters, call `train()` with your corpus of sentences,
    and then retrieve word or sentence embeddings as needed. This implementation is suitable for educational
    purposes and small-scale applications but may require enhancements for production use cases.

    ## Limitations
    While this implementation serves as a good starting point for understanding Word2Vec, it lacks several
    advanced features present in more robust libraries (e.g., Gensim). These include:

    - Negative sampling
    - Hierarchical softmax
    - Pre-trained embeddings support

    Users seeking high performance and scalability should consider leveraging established libraries or frameworks.

    Overall, this SimpleWord2Vec class provides an accessible entry point into natural language processing and
    machine learning concepts related to word embeddings.

    """

    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 50, window_size: int = 2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_count = {}
        self.embeddings = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes the input text into words."""
        return text.lower().split()

    def build_vocab(self, sentences: List[str]) -> None:
        """Builds vocabulary from the input sentences."""
        words = []
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            words.extend(tokens)
            for word in tokens:
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

        # Create word to index mapping
        unique_words = list(self.word_count.keys())
        for i, word in enumerate(unique_words):
            if i < self.vocab_size:
                self.word_to_index[word] = i
                self.index_to_word[i] = word

        # Initialize embeddings matrix
        self.embeddings = np.random.rand(len(self.word_to_index), self.embedding_dim)

    def _generate_training_data(self, sentences: List[str]) -> List[Tuple[int, List[int]]]:
        """Generates training data for the model."""
        training_data = []

        for sentence in sentences:
            tokens = self._tokenize(sentence)
            indices = [self.word_to_index[token] for token in tokens if token in self.word_to_index]

            for i, target in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                context_indices = [indices[j] for j in range(start, end) if j != i]
                training_data.append((target, context_indices))

        return training_data

    def train(self, sentences: List[str], epochs: int = 1000) -> None:
        """Trains the Word2Vec model on the given sentences."""
        self.build_vocab(sentences)
        training_data = self._generate_training_data(sentences)

        for epoch in range(epochs):
            loss = 0
            for target, contexts in training_data:
                target_vector = self.embeddings[target]

                # Update embeddings based on context words
                for context in contexts:
                    context_vector = self.embeddings[context]
                    similarity = np.dot(target_vector, context_vector)
                    error = similarity - 1  # Target is to predict 1 (similarity)
                    loss += error ** 2

                    # Update weights (gradient descent step)
                    grad = error * context_vector
                    self.embeddings[target] -= grad * 0.01  # Learning rate
                    self.embeddings[context] -= grad * 0.01

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def get_vector(self, word: str) -> np.ndarray:
        """Returns the vector representation of a word."""
        index = self.word_to_index.get(word)
        if index is not None:
            return self.embeddings[index]
        else:
            raise ValueError(f"Word '{word}' not found in vocabulary.")

    def _generate_sentence_embedding(self, sentence: str) -> np.ndarray:
        """
        Generate an embedding for a sentence using Word2Vec-like word embeddings.

        :param sentence: Input sentence.
        :return: Sentence-level embedding as a NumPy array.
        """
        tokens = self._tokenize(sentence)

        valid_vectors = [self.get_vector(token) for token in tokens if token in self.word_to_index]

        if not valid_vectors:
            raise ValueError("No valid tokens found in the sentence.")

        return np.mean(valid_vectors, axis=0)
