# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
import numpy as np

# from spacy import textacy
from spacy import displacy
from spacy.tokens.doc import Doc
from preprocess import Preprocessor
from collections import Counter


def num_tokens(doc: Doc):
    return len(doc)


def num_types(doc: Doc):
    word_frequencies = Counter()

    for sentence in doc.sents:
        words = [token.text for token in sentence if not token.is_punct]
        word_frequencies.update(words)

    return len(word_frequencies.keys())


def num_words(doc: Doc):  # Think about punctuation
    word_frequencies = Counter()

    for sentence in doc.sents:
        words = [token.text for token in sentence if not token.is_punct]
        word_frequencies.update(words)

    return sum(word_frequencies.values())


def avg_words_sentence(doc: Doc):
    word_count_sentences = []

    for sentence in doc.sents:
        word_frequencies = Counter()

        words = [token.text for token in sentence if not token.is_punct]
        word_frequencies.update(words)

        word_count_sentences.append(sum(word_frequencies.values()))

    return np.mean(word_count_sentences)


def avg_word_length(doc: Doc):
    word_lengths = []

    for sentence in doc.sents:
        words = [token.text for token in sentence if not token.is_punct]
        [word_lengths.append(len(word)) for word in words]

    return np.mean(word_lengths)


def token_bigrams(doc: Doc):
    bigram_frequencies = Counter()

    for i in range(len(doc) - 1):
        bigram = (doc[i].text, doc[i + 1].text)
        bigram_frequencies.update([bigram])

    return bigram_frequencies.most_common(3)


def token_trigrams(doc: Doc):
    trigram_frequencies = Counter()

    for i in range(len(doc) - 2):
        trigram = (doc[i].text, doc[i + 1].text, doc[i + 2].text)
        trigram_frequencies.update([trigram])

    return trigram_frequencies.most_common(3)


def pos_bigrams(doc: Doc):
    bigram_frequencies = Counter()

    for i in range(len(doc) - 1):
        bigram = (doc[i].pos_, doc[i + 1].pos_)
        bigram_frequencies.update([bigram])

    return bigram_frequencies.most_common(3)


def pos_trigrams(doc: Doc):
    trigram_frequencies = Counter()

    for i in range(len(doc) - 2):
        trigram = (doc[i].pos_, doc[i + 1].pos_, doc[i + 2].pos_)
        trigram_frequencies.update([trigram])

    return trigram_frequencies.most_common(3)


def n_named_entites(doc: Doc):
    return len(doc.ents)


def n_unique_labels(doc: Doc):
    return len(np.unique([ent.label_ for ent in doc.ents]))


def visualize(doc: Doc):
    first_five = list(doc.sents)[:5]

    displacy.serve(first_five, style="ent", port=5001)


def main():
    nlp = spacy.load("en_core_web_sm")

    with open("data/preprocessed/train/sentences.txt", "r") as text_file:
        text_data = text_file.read()

    preprocessor = Preprocessor(text_data)

    text_data = preprocessor.process()

    doc = nlp(text_data)

    print("Number of tokens:", num_tokens(doc))
    print("Number of types:", num_types(doc))
    print("Number of words:", num_words(doc))
    print("Average number of words per sentence:", round(avg_words_sentence(doc), 2))
    print("Average word length:", round(avg_word_length(doc), 2))

    print("\nToken bigrams:", token_bigrams(doc))
    print("Token trigrams:", token_trigrams(doc))

    print("\nPOS bigrams:", pos_bigrams(doc))
    print("POS trigrams:", pos_trigrams(doc))

    print("\nNumber of named entities:", n_named_entites(doc))
    print("Number of different entity labels:", n_unique_labels(doc))

    # Uncomment to visualize NER on first 5 sentences
    # visualize(doc)


if __name__ == "__main__":
    main()
