# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from re import L
from model.data_loader import DataLoader
from statistics import mode
import random
import numpy as np

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

POSITIVE_CLASS = "C"


def confusion_matrix(gold, predicted):
    matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for g, p in zip(gold, predicted):
        for a, b in zip(g, p):
            if b == POSITIVE_CLASS:
                if a == b:
                    matrix["TP"] += 1
                else:
                    matrix["FP"] += 1
            else:
                if a == b:
                    matrix["TN"] += 1
                else:
                    matrix["FN"] += 1

    return matrix


def precision(gold, predicted):
    matrix = confusion_matrix(gold, predicted)
    TP = matrix["TP"]
    FP = matrix["FP"]
    TN = matrix["TN"]
    FN = matrix["FN"]
    return TP / (TP + FP)


def recall(gold, predicted):
    matrix = confusion_matrix(gold, predicted)
    TP = matrix["TP"]
    FP = matrix["FP"]
    TN = matrix["TN"]
    FN = matrix["FN"]
    return TP / (TP + FN)


def accuracy_metric(gold, predicted):
    matrix = confusion_matrix(gold, predicted)
    TP = matrix["TP"]
    FP = matrix["FP"]
    TN = matrix["TN"]
    FN = matrix["FN"]
    return (TP + TN) / (TP + FP + TN + FN)


def majority_baseline(train_sentences, train_labels, test_input, test_labels):
    accuracy = None

    majority_class = mode(sum(train_labels, []))

    predictions = []
    for instance in test_input:
        instance_predictions = [majority_class for t in instance]
        predictions.append(instance_predictions)

    accuracy = accuracy_metric(test_labels, predictions)
    return accuracy, predictions


def random_baseline(train_sentences, train_labels, test_input, test_labels):
    predictions = []
    for instance in test_input:
        instance_predictions = [random.choice(["N", "C"]) for t in instance]
        predictions.append(instance_predictions)

    accuracy = accuracy_metric(test_labels, predictions)
    return accuracy, predictions


def length_baseline(train_sentences, train_labels, test_input, test_labels, k=1):
    predictions = []
    for instance in test_input:
        instance_predictions = ["C" if len(t) >= k else "N" for t in instance]
        predictions.append(instance_predictions)

    accuracy = accuracy_metric(test_labels, predictions)
    return accuracy, predictions


def frequency_baseline(train_sentences, train_labels, test_input, test_labels, freq=20):
    predictions = []
    joined = " ".join([" ".join(l) for l in test_input])
    length = len(joined.split())

    for instance in test_input:
        instance_predictions = [
            "C" if joined.count(t) / length <= freq else "N" for t in instance
        ]
        predictions.append(instance_predictions)

    accuracy = accuracy_metric(test_labels, predictions)
    return accuracy, predictions


def variate_length(train_sentences, train_labels, dev_sentences, dev_labels):
    best_k = 0
    max_accuracy = -1
    best_predictions = None

    for k in range(1, 30):
        accuracy, length_predictions = length_baseline(
            train_sentences, train_labels, test_sentences, test_labels, k=k
        )

        if accuracy >= max_accuracy:
            best_k = k
            max_accuracy = accuracy
            best_predictions = length_predictions
    return max_accuracy, best_predictions, best_k


def variate_frequency(train_sentences, train_labels, dev_sentences, dev_labels):
    best_freq = 0
    max_accuracy = -1
    best_predictions = None

    for freq in np.arange(0.01, 99.99, 0.01):
        accuracy, frequency_predictions = frequency_baseline(
            train_sentences, train_labels, test_sentences, test_labels, freq=freq
        )

        if accuracy >= max_accuracy:
            best_freq = freq
            max_accuracy = accuracy
            best_predictions = frequency_predictions
    return max_accuracy, best_predictions, best_freq


if __name__ == "__main__":
    random.seed(42)
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()
        train_sentences = [x.strip("\n").split() for x in train_sentences]
    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()
        train_labels = [x.strip("\n").split() for x in train_labels]

    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()
        dev_sentences = [x.strip("\n").split() for x in dev_sentences]
    with open(dev_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()
        dev_labels = [x.strip("\n").split() for x in dev_labels]

    with open(test_path + "sentences.txt") as testfile:
        test_sentences = testfile.readlines()
        test_sentences = [x.strip("\n").split() for x in test_sentences]
    with open(test_path + "labels.txt") as test_labelfile:
        test_labels = test_labelfile.readlines()
        test_labels = [x.strip("\n").split() for x in test_labels]

    majority_accuracy_dev, majority_predictions_dev = majority_baseline(
        train_sentences, train_labels, dev_sentences, dev_labels
    )
    majority_accuracy_test, majority_predictions_test = majority_baseline(
        train_sentences, train_labels, test_sentences, test_labels
    )
    print("MAJORITY baseline accuracy DEV:", round(majority_accuracy_dev, 2))
    print("MAJORITY baseline accuracy TEST:", round(majority_accuracy_test, 2))

    random_accuracy_dev, random_predictions_dev = random_baseline(
        train_sentences, train_labels, dev_sentences, dev_labels
    )
    random_accuracy_test, random_predictions_test = random_baseline(
        train_sentences, train_labels, test_sentences, test_labels
    )
    print("\nRANDOM baseline accuracy DEV:", round(random_accuracy_dev, 2))
    print("RANDOM baseline accuracy TEST:", round(random_accuracy_test, 2))

    length_accuracy_dev, length_predictions_dev, k = variate_length(
        train_sentences, train_labels, dev_sentences, dev_labels
    )
    length_accuracy_test, length_predictions_test = length_baseline(
        train_sentences, train_labels, test_sentences, test_labels, k=k
    )
    print(
        f"\nLENGTH ≥{k} baseline accuracy DEV:",
        round(length_accuracy_dev, 2),
    )
    print(
        f"LENGTH ≥{k} baseline accuracy TEST:",
        round(length_accuracy_test, 2),
    )

    freq_accuracy_dev, freq_predictions_dev, freq = variate_frequency(
        train_sentences, train_labels, dev_sentences, dev_labels
    )
    freq_accuracy_test, freq_predictions_test = frequency_baseline(
        train_sentences, train_labels, test_sentences, test_labels, freq=freq
    )
    print(
        f"\nFREQUENCY <={freq} baseline accuracy DEV:",
        round(freq_accuracy_dev, 2),
    )
    print(
        f"FREQUENCY <={freq} baseline accuracy TEST:",
        round(freq_accuracy_test, 2),
    )
