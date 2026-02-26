
import random
import PyPDF2
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')


with open("resources/US_Declaration.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + " "




sentences = nltk.sent_tokenize(text)


random_sentence = random.choice(sentences)
print('A random sentence:\n' + random_sentence + '\n')



from nltk.tokenize import word_tokenize

tokens = word_tokenize(random_sentence)
print('The tokens:\n' + str(tokens))


from nltk.stem import PorterStemmer


stemmer = PorterStemmer()
stems = []
for word in tokens:
    stems += [stemmer.stem(word)]


print('The stem words:\n' + str(stems))



# 1. New downloads required
nltk.download('wordnet')
nltk.download('stopwords')

# --- STOP WORDS REMOVAL ---
from nltk.corpus import stopwords

# Use a set for faster lookups
stop_words = set(stopwords.words('english'))

# Filter out stop words (e.g., 'the', 'is', 'in')
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]

print('Tokens without stop words:\n' + str(filtered_tokens))


# --- LEMMATIZATION ---
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmas = []

# Lemmatize the filtered tokens
for word in filtered_tokens:
    # Note: WordNetLemmatizer defaults to Noun. 
    # For better accuracy with verbs, use lemmatizer.lemmatize(word, pos='v')
    lemmas.append(lemmatizer.lemmatize(word))

print('The lemmatized words:\n' + str(lemmas))
