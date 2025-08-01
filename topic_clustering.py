import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import warnings
import psycopg2

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


# LDA
lda_keywords_df = get_topic_words(lda, count_vectorizer.get_feature_names_out(), 15)
lda_keywords_df.to_csv('lda_topics_keywords.csv', index=False)

lda_topic_distributions = lda.transform(count_data)
df['LDA_topic'] = lda_topic_distributions.argmax(axis=1) + 1  # Topics 1-indexed
df['LDA_score'] = lda_topic_distributions.max(axis=1)

df[['title', 'published_at', 'LDA_topic', 'LDA_score']].to_csv('headlines_with_lda_topics.csv', index=False)

# NMF 
nmf_keywords_df = get_topic_words(nmf, tfidf_vectorizer.get_feature_names_out(), 15)
nmf_keywords_df.to_csv('nmf_topics_keywords.csv', index=False)

nmf_topic_distributions = nmf.transform(tfidf_data)
df['NMF_topic'] = nmf_topic_distributions.argmax(axis=1) + 1  # 1-indexed
df['NMF_score'] = nmf_topic_distributions.max(axis=1)

df[['title', 'published_at', 'NMF_topic', 'NMF_score']].to_csv('headlines_with_nmf_topics.csv', index=False)
