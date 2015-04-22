### Utilities:
import numpy as np
try:
    from xml_cleaner import to_raw_text
except ImportError:
    def to_raw_text(text):
        raise ImportError("Could not import xml_cleaner. Try `pip3 install xml_cleaner`.")

def tokenize(text):
    return [word for sentence in to_raw_text(text) for word in sentence]

from collections import Counter

class Vocab:
    __slots__ = ["word2index", "index2word", "unknown", "word_occ"]

    def __init__(self, index2word = None):
        self.word2index = {}
        self.word_occ = Counter()
        self.index2word = []

        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0

        if index2word is not None:
            self.add_words(index2word)

    def add_words(self, words):
        self.word_occ.update(words)
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

    def add_words_with_min_occurences(self, words, threshold):
        if threshold == 0:
            return add_words(words)
        for word in words:
            self.word_occ[word] += 1
            if self.word_occ[word] >= threshold and (word not in self.word2index):
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

    def add_words_from_text(self, text, tokenization=False, threshold=5):
        tokens = text.split(" ") if not tokenization else tokenize(text)
        self.add_words_with_min_occurences(tokens, threshold)

    def __getitem__ (self, word):
        return self.word2index[word]

    def __contains__(self, word):
        return word in self.word2index

    def __call__(self, line, tokenization = False):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ") if not tokenization else tokenize(line)
            indices = np.zeros(len(line), dtype=np.int32)

        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)

        return indices

    def __sub__(self, other):
        assert(type(other) == type(self)), "can only subtract Vocabs together"
        delta_vocab = Vocab(list(set(self.index2word) - set(other.index2word)))

        for word in delta_vocab.index2word:
            if word in other:
                delta_vocab.word_occ[word] = other.word_occ[word]
            else:
                delta_vocab.word_occ[word] = self.word_occ[word]
        return delta_vocab

    @property
    def size(self):
        return len(self.index2word)

    def __len__(self):
        return len(self.index2word)
