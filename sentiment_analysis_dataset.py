import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Import the kaggle dataset
file_path = 'Twitter_Data.csv' 
df = pd.read_csv(file_path)
df = df.head(10)
# Preprocessing the tweet function
def preprocess_tweet(tweet):
    # Check if the tweet is a string
    if isinstance(tweet, str):  
        tweet_words = []
        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = 'http'
            tweet_words.append(word)
        return ' '.join(tweet_words)
    else:
        return ""  
        # Return an empty string if the tweet is not a string

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ["Negative", "Neutral", "Positive"]

# Sentiment analysis function
def analyze_sentiment(tweet):
    tweet_proc = preprocess_tweet(tweet)
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return labels[scores.argmax()]

# Apply sentiment analysis to each tweet in the clean_text column
df['predicted_sentiment'] = df['clean_text'].apply(analyze_sentiment)

# Save the resulting DataFrame to a new CSV file
df.to_csv('new_data.csv', index=False)

# Display the first few rows of the updated DataFrame
print(df.head())
