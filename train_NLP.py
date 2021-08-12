# import required packages
from utils import *


if __name__ == "__main__":
    # datasetDir = getIMDBData()  # Utility to download and unzip dataset
    removeRedundantDirectory()

    # Loading training data from train directory
    trainSet = text_dataset_from_directory(
        'aclImdb/train', batch_size=300,
        seed=100, label_mode='int')

    # Loading test data for validation from train directory
    testSet = text_dataset_from_directory(
        'aclImdb/test', label_mode='int')

    # Word preprocessing rules/predicates
    wordVectorizePredicates = getVectorizationPredicates()
    textSet = trainSet.map(lambda texts, labels: texts)
    wordVectorizePredicates.adapt(textSet)

    # Configuring the best model out of all use cases
    SentimentModel = Sequential([
        wordVectorizePredicates,
        Embedding(vocabularies, embeddingDimension, name="embedding"),
        Conv1D(filters=40, kernel_size=3, padding='same', activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(20, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compiling the model
    SentimentModel.compile(optimizer='adam',
                           loss=BinaryCrossentropy(from_logits=False),
                           metrics=['accuracy'])

    # Callback function to save on epochs completion
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train the network and save checkpoint to load later
    history = SentimentModel.fit(
        trainSet,
        validation_data=testSet,
        epochs=2, callbacks=[cp_callback])
    scores = SentimentModel.evaluate(testSet, verbose=0)

    # Reporting Metrics and Model configuration
    print("var_loss on validation set: %.2f%%" % (scores[0]))
    print("Accuracy on validation set: %.2f%%" % (scores[1] * 100))

    print(SentimentModel.summary())
    plotAccuracyVsEpochs(history)
    plotLossVsEpochs(history)