# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    # TODO: determine the majority class based on the training data
    # ...
    majority_class = "X"
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance, instance_predictions)

    # TODO: calculate accuracy for the test input
    # ...
    return accuracy, predictions





if __name__ == '__main__':
    train_path = "data/preprocessed/train"
    dev_path = "data/preprocessed/dev"
    test_path = "data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()


    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(train_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()
    with open(test_path + "sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open("test_path + labels.txt") as test_labelfile:
        testlabels = test_label_file.readlines()
    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, testinput)

    # TODO: output the predictions in a suitable way so that you can evaluate them