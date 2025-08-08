import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import warnings
import psycopg2
import matplotlib.pyplot as plt
from prophet import Prophet

warnings.filterwarnings('ignore')
nltk.download('stopwords')

conn = psycopg2.connect("host=localhost dbname=newsdb user=newadmin password=admin")
df = pd.read_sql("SELECT id, title, published_at FROM news_articles", conn)
conn.close()


# === 2. Preprocessing ===
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = [word.lower() for word in text.split() if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_title'] = df['title'].apply(preprocess)

# === 3. Vectorization ===
n_topics = 3

# For LDA
count_vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
count_data = count_vectorizer.fit_transform(df['clean_title'])

# For NMF
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, stop_words='english')
tfidf_data = tfidf_vectorizer.fit_transform(df['clean_title'])

# === 4. LDA Model ===
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(count_data)

# === 5. NMF Model ===
nmf = NMF(n_components=n_topics, random_state=42)
nmf.fit(tfidf_data)

# === 6. Display Topics Function ===
def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx + 1}: ", " | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def get_topic_words(model, feature_names, num_words=10):
    topic_keywords = []
    for idx, topic in enumerate(model.components_):
        keywords = [(feature_names[i], topic[i]) for i in topic.argsort()[:-num_words-1:-1]]
        for word, weight in keywords:
            topic_keywords.append({'topic': f'Topic {idx+1}', 'word': word, 'weight': weight})
    return pd.DataFrame(topic_keywords)

def export_topic_forecasts(trend_df, topic_numbers, forecast_days=14, output_filename='topics_forecast.xlsx'):
    """
    Fits Prophet models and exports forecasts for specified topics into an Excel file (one sheet per topic).
    
    Parameters:
    -----------
    trend_df : pd.DataFrame
        DataFrame with columns ['published_at', 'NMF_topic', 'count'].
    topic_numbers : list
        List of topic numbers to forecast.
    forecast_days : int
        Number of days to forecast into the future.
    output_filename : str
        Name of the output Excel file.
    """
    forecast_dfs = {}

    # Identify all *_topic columns
    topic_cols = [col for col in trend_df.columns if col.endswith('_topic')]

    # Create a unified 'topic' column (first non-null per row)
    trend_df['topic'] = trend_df[topic_cols].bfill(axis=1).iloc[:, 0]

    for topic in topic_numbers:
        
        # Prepare the data for the current topic
        topic_data = trend_df[trend_df['topic'] == topic][['published_at', 'count']].copy()
        topic_data = topic_data.rename(columns={'published_at': 'ds', 'count': 'y'})

        # Check if there's enough data to fit Prophet
        if topic_data.shape[0] < 2:
            print(f"Not enough data for topic {topic}, skipping.")
            continue

        # Fit Prophet model
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)
        m.fit(topic_data)

        # Forecast
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_export.columns = ['Date', 'Predicted_Count', 'Lower_Bound', 'Upper_Bound']
        forecast_export.reset_index(drop=True, inplace=True)

        # Save to dictionary for later export
        forecast_dfs[f'Topic_{topic}'] = forecast_export

    # Write all to Excel
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for sheet_name, df_topic in forecast_dfs.items():
            df_topic.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Export completed: {output_filename}")

# LDA
lda_keywords_df = get_topic_words(lda, count_vectorizer.get_feature_names_out(), 15)
lda_keywords_df.to_csv('lda_topics_keywords.csv', index=False)

lda_topic_distributions = lda.transform(count_data)
df['LDA_topic'] = lda_topic_distributions.argmax(axis=1) + 1  # Topics 1-indexed
df['LDA_score'] = lda_topic_distributions.max(axis=1)

df[['title', 'published_at', 'LDA_topic', 'LDA_score']].to_csv('headlines_with_lda_topics.csv', index=False)
trend_df = df.groupby(['published_at', 'LDA_topic']).size().reset_index(name='count')
export_topic_forecasts(trend_df, [1, 2, 3], output_filename='lda_topic_forecast.xlsx')

# NMF 
nmf_keywords_df = get_topic_words(nmf, tfidf_vectorizer.get_feature_names_out(), 15)
nmf_keywords_df.to_csv('nmf_topics_keywords.csv', index=False)

nmf_topic_distributions = nmf.transform(tfidf_data)
df['NMF_topic'] = nmf_topic_distributions.argmax(axis=1) + 1  # 1-indexed
df['NMF_score'] = nmf_topic_distributions.max(axis=1)

df[['title', 'published_at', 'NMF_topic', 'NMF_score']].to_csv('headlines_with_nmf_topics.csv', index=False)

trend_df = df.groupby(['published_at', 'NMF_topic']).size().reset_index(name='count')
export_topic_forecasts(trend_df, [1, 2, 3], output_filename='nmf_topic_forecast.xlsx')