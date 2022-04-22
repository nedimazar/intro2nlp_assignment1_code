# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
class Evaluation:
    def __init__(
        self,
        model_output_file="experiments/base_model/model_output.tsv",
        positive_class="C",
    ):
        with open(model_output_file, "r") as data_file:
            self.data = data_file.readlines()
        self.positive_class = positive_class
        self.build_confusion_matrix()

    def build_confusion_matrix(self):
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0

        for line in self.data:
            split = line.split()
            if len(split) == 3:
                word = split[0]
                gold = split[1]
                prediction = split[2]

                if prediction == self.positive_class:
                    if gold == prediction:
                        self.TP += 1
                    else:
                        self.FP += 1
                else:
                    if gold == prediction:
                        self.TN += 1
                    else:
                        self.FN += 1

    def precision(self):
        try:
            return self.TP / (self.TP + self.FP)
        except ZeroDivisionError:
            return 0

    def recall(self):
        try:
            return self.TP / (self.TP + self.FN)
        except ZeroDivisionError:
            return 0

    def F1(self):
        try:
            return 2 * (
                (self.precision() * self.recall()) / (self.precision() + self.recall())
            )
        except ZeroDivisionError:
            return 0

    def print_metrics(self):
        print(f"Class {self.positive_class}")
        print(f"\t-- Precision : {round(self.precision(), 2)}")
        print(f"\t-- Recall    : {round(self.recall(), 2)}")
        print(f"\t-- F1        : {round(self.F1(), 2)}")


def main():
    e1 = Evaluation()
    e1.print_metrics()
    F1C = e1.F1()

    e2 = Evaluation(positive_class="N")
    e2.print_metrics()
    F1N = e2.F1()

    wavg = (350 * F1N + 89 * F1C) / (350 + 89)
    print("Weighted average F1:", round(wavg, 2))


if __name__ == "__main__":
    main()
