from collections import defaultdict, Counter
import random
import re


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        # corpus 是 main.py 传进来的句子列表，先拼成一整段文本
        text = " ".join(corpus)
        words = self._simple_tokenizer(text, frequency_threshold)
        self.vocab, self.bigram_probs = self._build_bigrams(words)

    def _simple_tokenizer(self, text, frequency_threshold=5):
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        word_counts = Counter(tokens)
        return [t for t in tokens if word_counts[t] >= frequency_threshold]

    def _build_bigrams(self, words):
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            bigram_probs[w1][w2] = count / unigram_counts[w1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:
                break
            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values()
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)