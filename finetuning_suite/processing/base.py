from abc import ABC, abstractmethod

class Preprocessor(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_function(self, sample, padding="max_length"):
        pass


# Step 2: Default Preprocessor
class DefaultPreprocessor(Preprocessor):

    def preprocess_function(self, sample, padding="max_length", max_source_length=None, max_target_length=None):
        inputs = ["summarize" + item for item in sample["dialogue"]]
        model_inputs = self.tokenizer(inputs, max_length=max_source_length, truncation=True)
        labels = self.tokenizer(text_target=sample["summary"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs