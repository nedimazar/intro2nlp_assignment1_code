# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
from collections import Counter


def num_tokens(doc):
    return len(doc)


def num_types(doc):
    word_frequencies = Counter()

    for sentence in doc.sents:
        words = [token.text for token in sentence if not token.is_punct]
        word_frequencies.update(words)

    return len(word_frequencies.keys())


def num_words(doc):
    return -1


def avg_words_sentence(doc):
    return -1


def avg_word_length(doc):
    return -1


def main():
    nlp = spacy.load("en_core_web_sm")

    with open("data/preprocessed/train/sentences.txt", "r") as text_file:
        text_data = text_file.read()

    doc = nlp(text_data)

    print("Number of tokens:", num_tokens(doc))
    print("Number of types:", num_types(doc))
    print("Number of words:", num_words(doc))
    print("Average number of words per sentence:", round(avg_words_sentence(doc), 2))
    print("Average word length:", round(avg_word_length(doc), 2))


if __name__ == "__main__":
    main()
