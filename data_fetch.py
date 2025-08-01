import psycopg2
import os

from dotenv import load_dotenv
from datetime import datetime
from newsapi import NewsApiClient
from datetime import datetime, timedelta

load_dotenv()

# ==== Config ====
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
COUNTRY = "us"  # change as needed

PG_HOST = "localhost"
PG_DB = "newsdb"
PG_USER = "newadmin"
PG_PASS = "admin"
PG_PORT = 5432

def fetch_news():
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    day_before_yesterday = datetime.now() - timedelta(days=7)
    next_day = datetime.now() - timedelta(days=1)
    from_param = day_before_yesterday.strftime('%Y-%m-%d')
    to_param = next_day.strftime('%Y-%m-%d')

    all_articles = newsapi.get_everything(
        q='trade deal',
        from_param=from_param,
        to=to_param,
        language='en',
        sort_by='publishedAt'
    )

    return all_articles["articles"]

# ==== Save to PostgreSQL ====
def save_articles_to_db(articles):
    conn = psycopg2.connect(
        host=PG_HOST,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASS,
        port=PG_PORT
    )
    cur = conn.cursor()

    insert_query = """
        INSERT INTO news_articles
        (source_name, author, title, description, url, published_at, content)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING;
    """

    for a in articles:
        cur.execute(insert_query, (
            a['source']['name'],
            a.get('author'),
            a.get('title'),
            a.get('description'),
            a.get('url'),
            datetime.fromisoformat(a['publishedAt'].replace("Z", "+00:00")),
            a.get('content')
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(articles)} articles.")

# ==== Main Flow ====
if __name__ == "__main__":
    news_articles = fetch_news()
    save_articles_to_db(news_articles)
