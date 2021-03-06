
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
import re

class TextPreprocessor(object):
    def _clean_line(self, text):
        """
        _clean_line will clean the text by removing special caratere and web link, reformat the text ( lower case )
         Args:
            text(string ): string of the text to clean.
        Returns:
            text (String): cleaned text.
        """
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@[A-Za-z0-9]+", "", text)
        text = re.sub(r"#[A-Za-z0-9]+", "", text)
        text = text.replace("RT","")
        text = text.lower()
        text = text.strip()
        return text
    
    def __init__(self, vocab_size, max_sequence_length):
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        self._tokenizer = None

    def fit(self, text_list):
        """
        fit a tokenizer model of the class to list of string
         Args:
            text_list(List ): List of string to feed to the model.
        """
        # Create vocabulary from input corpus.
        text_list_cleaned = [self._clean_line(txt) for txt in text_list]
        tokenizer = text.Tokenizer(num_words=self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer = tokenizer

    def transform(self, text_list):
        """
               transform will clean and tokenize a list of text
                Args:
                   text_list(List ): List of string to transform
               Returns:
                   padded_text_sequence (List): transformed text.
               """
        # Transform text to sequence of integers
        text_list = [self._clean_line(txt) for txt in text_list]
        text_sequence = self._tokenizer.texts_to_sequences(text_list)

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        padded_text_sequence = sequence.pad_sequences(
          text_sequence, maxlen=self._max_sequence_length)
        return padded_text_sequence
