import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

text = "Hello! How are you? I am fine."

print(sent_tokenize(text))