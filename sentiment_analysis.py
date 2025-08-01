import psycopg2
import pandas as pd
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Choose emotion detection model by argument")
parser.add_argument("model_choice", type=int, help="Enter 1 for j-hartmann model, anything else for cardiffnlp model")
args = parser.parse_args()

# 1. Load Data
conn = psycopg2.connect("host=localhost dbname=newsdb user=newadmin password=admin")
df = pd.read_sql("SELECT id, title, published_at FROM news_articles", conn)
conn.close()

# Decide model name based on argument
if args.model_choice == 1:
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    exported_file = 'news_sentiment_per_day_jhartmann.xlsx'
else:
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    exported_file = 'news_sentiment_per_day_cardiffnlp.xlsx'


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = model.config.id2label

# 3. Predict Emotions
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs).item()
    return labels[pred_idx]

tqdm.pandas()
df['emotion'] = df['title'].progress_apply(predict_emotion)

# 4. Map to Custom Categories
emotion_map = {
    'joy': 'positive',
    'excitement': 'positive',
    'optimism': 'positive',
    'admiration': 'positive',
    'pride': 'positive',
    'sadness': 'negative',
    'fear': 'negative',
    'nervousness': 'negative',
    'anger': 'negative',
    'disgust': 'negative',
    'remorse': 'negative',
    'embarrassment': 'negative',
    'approval': 'positive',
    'disappointment': 'negative',
    'neutral': 'neutral',
}
df['category'] = df['emotion'].map(lambda x: emotion_map.get(x, 'other'))

# 5. Aggregate Per Day
df['date'] = pd.to_datetime(df['published_at']).dt.date
daily_sentiment = df.groupby(['date', 'category']).size().unstack(fill_value=0)
print(daily_sentiment)
daily_sentiment.to_excel(exported_file)
