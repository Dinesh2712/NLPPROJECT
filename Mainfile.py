import pandas as pd
import re
import string
import collections
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Dense, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


df_train = pd.read_csv(r"C:\Users\Don\Downloads\archive (6)\train.csv")
df_test = pd.read_csv(r"C:\Users\Don\Downloads\archive (6)\test.csv")

df_train.columns = ['label', 'title', 'text']
df_train.head()

df_test.columns = ['label', 'title', 'text']
df_test.head()

new_df_train = df_train.head(50000)
new_df_test = df_test.head(10000)


def concat_columns(df, col1, col2, new_col):
    df[new_col] = df[col1].apply(str) + ' ' + df[col2].apply(str)
    df.drop(col2, axis = 1, inplace = True)
    return df

new_df_train = concat_columns(new_df_train, 'text', 'title', 'text')
new_df_test = concat_columns(new_df_test, 'text', 'title', 'text')

new_df_train['label'] = df_train['label'].map({1:0, 2:1})
new_df_test['label'] = df_test['label'].map({1:0, 2:1})

def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-zÀ-ú ]+', '', text)
    # Removing repetitive words
    text = re.sub('book|one', '', text)
    # Convert to lower case
    text = text.lower()
    # remove scores
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

new_df_train['text'] = new_df_train['text'].apply(clean_text)
new_df_test['text'] = new_df_test['text'].apply(clean_text)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([word for word in tokens if word not in (stop_words)])

new_df_train['text'] = new_df_train['text'].apply(remove_stopwords)
new_df_test['text'] = new_df_test['text'].apply(remove_stopwords)

words = []
for text in new_df_test['text']:
    words.extend(text.split())
word_count = collections.Counter(words)
top_words = dict(word_count.most_common(10))

# Figure Size
plt.figure(figsize = (10, 6))

# Create the Barplot
plt.bar(range(len(top_words)), list(top_words.values()), align = 'center')

# Creating a y axis with words
plt.xticks(range(len(top_words)), list(top_words.keys()))

# Grid Opacity
plt.grid(alpha = 0.5)

# Title and labels
plt.title('Top 10 most used words', fontsize = 18)
plt.xlabel('Words')
plt.ylabel('Frequency')

# Maximum number of words to be considered in the vocabulary
max_words = 10000
# Maximum number of tokens in a sequence
max_len = 200
# Tokenizer
tokenizer = Tokenizer(num_words = max_words)
# Snap tokenizer to text data
tokenizer.fit_on_texts(df_train['text'])
# Converts texts into strings of numbers
sequences_train = tokenizer.texts_to_sequences(new_df_train['text'])
sequences_val = tokenizer.texts_to_sequences(new_df_test['text'])
# Mapping words to indexes
word_index = tokenizer.word_index

data_train = pad_sequences(sequences_train, maxlen = max_len)
data_val = pad_sequences(sequences_val, maxlen = max_len)


# Creating the model
model = tf.keras.Sequential()
model.add(Embedding(max_words, 16, input_length = max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Checking summary
model.summary()

history = model.fit(data_train, new_df_train['label'], epochs = 25, batch_size = 64, validation_data = (data_val, new_df_test['label']))

loss, accuracy = model.evaluate(data_val, new_df_test['label'], verbose = 0)
print('Accuracy: %f' % (accuracy*100))

# Total Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc = 'upper right')
plt.show()

# Total Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc = 'lower right')
plt.show()

from sklearn.metrics import classification_report

prediction = model.predict(data_val, verbose = 0)
rounded_integer_array = (np.rint(prediction)).astype(int)
print(classification_report(new_df_test['label'], rounded_integer_array))