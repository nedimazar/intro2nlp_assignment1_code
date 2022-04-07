# A simple class for the preprocessing of text input for NLP tasks
# For English
# We did not end up using this.


class Preprocessor:
    text_data: str

    def __init__(self, text_data):
        self.text_data = text_data

    def __periods(self):
        self.text_data = self.text_data.replace(" .", ".")

    def __commas(self):
        self.text_data = self.text_data.replace(" ,", ",")

    def __semicolons(self):
        self.text_data = self.text_data.replace(" ;", ";")

    def __colons(self):
        self.text_data = self.text_data.replace(" :", ":")

    def __backslashes(self):
        self.text_data = self.text_data.replace("\\", "")

    def __newlines(self):
        self.text_data = self.text_data.replace("\n", " ")

    def __parentheses(self):
        self.text_data = self.text_data.replace("( ", "(")
        self.text_data = self.text_data.replace(" )", ")")

    def __exclamations(self):
        self.text_data = self.text_data.replace(" !", "!")

    def __questions(self):
        self.text_data = self.text_data.replace(" ?", "?")

    def process(self):
        self.__periods()
        self.__commas()
        self.__semicolons()
        self.__colons()
        self.__backslashes()
        self.__newlines()
        self.__parentheses()
        self.__exclamations()
        self.__questions()

        return self.text_data
