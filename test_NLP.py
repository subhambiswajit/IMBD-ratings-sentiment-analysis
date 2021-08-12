# import required packages
from utils import *


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    # Load test set
    testSet = text_dataset_from_directory(
        'aclImdb/test', label_mode='int')

    # Load your saved model
    SentimentModel = getModel()

    SentimentModel.load_weights(checkpoint_path)

    # Run prediction on the test data and output required plot and loss
    scores = SentimentModel.evaluate(testSet, verbose=0)
    print("Loss on test Set: %.2f%%" % (scores[0]))
    print("Accuracy on test Set: %.2f%%" % (scores[1] * 100))