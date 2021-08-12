import os
import tensorflow as tf
import re
import string
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from keras.layers import Flatten, Conv1D
from tensorflow.keras.layers import Dropout

## Utilities related to NLP training

# No vocabularies considered
vocabularies = 10000
# Length cap of words in each review
lengthCapOfWords = 250
# Embedding vector/ feature representation size
embeddingDimension = 500
# Path of weights saved to and loaded from
checkpoint_path = "models/group53_NLP_model/group53_nlp_model.ckpt"

# Returns vectorized token of words after applying preprocessing predicates
def getVectorizationPredicates():
    wordVectorizePredicates = TextVectorization(
        standardize=getWordPrepocessingPredicates,
        max_tokens=vocabularies,
        output_mode='int',
        output_sequence_length=lengthCapOfWords)
    return wordVectorizePredicates

# Returns generic model decided for this task
def getModel():
    SentimentModel = Sequential([
        getVectorizationPredicates(),
        Embedding(vocabularies, embeddingDimension, name="embedding"),
        Conv1D(filters=40, kernel_size=3, padding='same', activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(20, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    SentimentModel.compile(optimizer='adam',
                           loss=BinaryCrossentropy(from_logits=False),
                           metrics=['accuracy'])
    return SentimentModel

# Syncs dataset
def getIMDBData():
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz",
                                      url,
                                      untar=True,
                                      cache_dir='.',
                                      cache_subdir='')
    return os.path.join(os.path.dirname(dataset), 'aclImdb')

# Removes unsup directory
def removeRedundantDirectory():
    trainDir = os.path.join('aclImdb', 'train')
    unsupDir = os.path.join(trainDir, 'unsup')
    if os.path.isdir(unsupDir):
        shutil.rmtree(unsupDir)

# Returns lowercase string
def makeTextsLowerCase(inputData):
    return tf.strings.lower(inputData)

# unnecessary br tags are observed in dataset
# Returns strings after removing HTML br tags
def removeHTMLTags(inputData):
    return tf.strings.regex_replace(inputData, '<br />',
                                    ' ')

# Returns strings after removing punctuations
def removePunctuations(inputData):
    return tf.strings.regex_replace(inputData,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

# Return the rules/ predicates for word processing
def getWordPrepocessingPredicates(inputData):
    processedData = makeTextsLowerCase(inputData)
    processedData = removePunctuations(processedData)
    processedData = removeHTMLTags(processedData)
    return processedData

# Plot model accuracy Vs epochs
def plotAccuracyVsEpochs(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Plot Loss vs epochs
def plotLossVsEpochs(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()