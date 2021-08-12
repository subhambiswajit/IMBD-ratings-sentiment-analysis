# IMBD-ratings-sentiment-analysis
Sentiment analysis of IMDB ratings using tensorflow Word2Vec 

![Iris image](images/imdb.png)

The IMDb Movie Reviews dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. The project downsyncs the dataset and no need of manually adding it to the project 

Framework used: Tensorflow <br>
Network topology: 

```
SentimentModel = Sequential([
        wordVectorizePredicates,
        Embedding(vocabularies, embeddingDimension, name="embedding"),
        Conv1D(filters=40, kernel_size=3, padding='same', activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(20, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
```
Accuracy achieved: 87%

### Activities taken:
1. String preprocessing
2. Word vectorisation
3. Generating word embeddings
4. Classification of a reviews from test folder as good or bad 
5. Saving the weights with a checkpoint

### How do i run the projects ? <br>
1. Run  ``` python train_NLP.py ``` to downsync dataset and train the network
2. Run  ``` python test_NLP.py ``` to predict the sentiments of reviews from the 
test folder.

### Info about other files <br>
1. utils.py has all the logic for 
    - Downsyncing the dataset
    - Preprocessing the data
    - Training the network
    - Saving checkpoint of trained weights
    
2. IMDBRating_sentiment_analysis.ipynb has all the attempts done for reaching the best network for the classification
