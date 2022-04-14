# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
import numpy as np
from tabulate import tabulate

# from spacy import textacy
from spacy import displacy
from spacy.tokens.doc import Doc
from collections import Counter
import pandas as pd


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


def lemmas(doc: Doc):
    lemma_dict = dict()

    for sent, i in zip(doc.sents, range(len(list(doc.sents)))):
        for token in sent:
            lemma = token.lemma_
            text = token.text

            if lemma not in lemma_dict:
                lemma_dict[lemma] = {text: {i}}
            else:
                if text not in lemma_dict[lemma]:
                    lemma_dict[lemma][text] = {i}
                else:
                    lemma_dict[lemma][text].add(i)

    print("A lemma that appears in more than 2 forms:")
    for lemma in lemma_dict:
        if lemma == "murder":
            print("***", lemma)
            for inflection in lemma_dict[lemma]:
                print("\t***", inflection)
                for i in lemma_dict[lemma][inflection]:
                    print("\t\t", list(doc.sents)[i])
                    break


def counter_to_relative(counter):
    """
    get relative frequencies of counter
    """
    total_count = sum(counter.values())
    relative = {}
    for key in counter:
        relative[key] = counter[key] / total_count
    return relative


def part_of_the_speech(doc):

    pos_frequencies = Counter()
    pos_tags_dict = {}
    pos_tags = []

    for sentence in doc.sents:
        # pos_tags = [token.pos_ for token in sentence]
        for token in sentence:
            tokens = []
            pos_tags.append(token.pos_)
            tokens.append(token.text)

            if not token.pos_ in pos_tags_dict.keys():
                pos_tags_dict[token.pos_] = Counter()

            pos_tags_dict[token.pos_].update(tokens)

    pos_frequencies.update(pos_tags)
    relative_freq = counter_to_relative(pos_frequencies)

    # print(pos_tags_dict)
    # print("most common verbs: \n",pos_tags_dict['VERB'].most_common(3),"\n")
    # print("less common verbs: \n",pos_tags_dict['VERB'].most_common()[-1],"\n")
    # print("pos_frecuencies:\n",pos_frequencies,"\n")
    # print("relative freq:\n",relative_freq,"\n")
    # print("sum: \n",sum(pos_frequencies.values()))

    return tabulate(
        [
            [
                pos,
                pos_frequencies[pos],
                round(relative_freq[pos], 2),
                pos_tags_dict[pos].most_common(3),
                pos_tags_dict[pos].most_common()[-1],
            ]
            for pos in pos_tags_dict.keys()
        ],
        headers=[
            "Finegrained POS-tag",
            "Occurrences",
            "Relative Tag Frequency (%)",
            "3 Most frequent tokens",
            "Infrequent token",
        ],
    )


def extract_basic_statistics():

    df = pd.read_csv("data/original/english/WikiNews_Train.tsv", sep="\t", header=None)

    # print(df.columns)

    # print(df.iloc[1:5, 10])

    # Number of instances labeled with 0:
    print("The number of instances labeled with 0: ", len(df.loc[df.iloc[:, 9] == 0]))

    # Number of instances labeled with 1:
    print("\nThe number of instances labeled with 1: ", len(df.loc[df.iloc[:, 9] == 1]))

    # Min, max, median, mean, and stdev of the probabilistic label
    print("\nProbabilistic Label:")
    print("\t max=", df.iloc[:, 10].max())
    print("\t min=", df.iloc[:, 10].min())
    print("\t median=", df.iloc[:, 10].median())
    print("\t mean=", round(df.iloc[:, 10].mean(), 2))
    print("\t stdev=", round(df.iloc[:, 10].std(), 2))

    # TODO
    # We probably need to use the Spacy tokenizer so it will count punctuation
    # Number of instances consisting of more than one token
    mask = df.iloc[:, 1].str.strip().str.split(" ").str.len()
    print(
        "\nNumber of instances consisting of more than one token: ", len(df[mask > 1])
    )

    # TODO
    # Same thing here with the tokenizer
    # Maximum number of tokens for an instance
    max_length = df.iloc[:, 1].str.strip().str.split(" ").str.len().max()
    print("\nMaximum number of tokens for an instance: ", max_length, "\t instance: ")
    print(
        str(
            df.iloc[:, 1][
                df.iloc[:, 1].str.strip().str.split(" ").str.len() == max_length
            ].iloc[:1]
        )
    )


def main():
    nlp = spacy.load("en_core_web_sm")

    with open(
        "data/preprocessed/train/sentences.txt", "r", encoding="utf8"
    ) as text_file:
        text_data = text_file.read()

    # Replacing newline characters with a space, otherwise the newline character becomes a very common token
    text_data = text_data.replace("\n", " ")
    # Getting rid of the escape backslash, because we do not think it is a part of Natural Language
    text_data = text_data.replace("\\", "")

    doc = nlp(text_data)

    # print("Number of tokens:", num_tokens(doc))
    # print("Number of types:", num_types(doc))
    # print("Number of words:", num_words(doc))
    # print("Average number of words per sentence:", round(avg_words_sentence(doc), 2))
    # print("Average word length:", round(avg_word_length(doc), 2))

    # print("\nToken bigrams:", token_bigrams(doc))
    # print("Token trigrams:", token_trigrams(doc))

    # print("\nPOS bigrams:", pos_bigrams(doc))
    # print("POS trigrams:", pos_trigrams(doc))

    # print("\nNumber of named entities:", n_named_entites(doc))
    # print("Number of different entity labels:", n_unique_labels(doc))

    # # Uncomment to visualize NER on first 5 sentences
    # # visualize(doc)

    # lemmas(doc)

    # print("Word Classes - Most frequent POS tags: \n",part_of_the_speech(doc))

    extract_basic_statistics()


if __name__ == "__main__":
    main()
