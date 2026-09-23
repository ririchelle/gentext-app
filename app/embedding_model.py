import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        self.nlp = spacy.load(model_name)

    def get_embedding(self, word):
        """Return the word's embedding vector as a plain Python list."""
        doc = self.nlp(word)
        return doc.vector.tolist()