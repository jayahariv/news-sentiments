# News Sentiment Analyzer

This Python project fetches the latest news headlines from [NewsAPI](https://newsapi.org/) and analyzes the sentiment of each title using state-of-the-art transformer models. It supports two emotion detection algorithms:

- [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- [`cardiffnlp/twitter-roberta-base-emotion`](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)

## Features

- Fetches live news headlines via NewsAPI.
- Analyzes headline sentiment/emotion using selected transformer model.
- Easily switch between two pre-trained HuggingFace models.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/news-sentiment-analyzer.git
   cd news-sentiment-analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get your NewsAPI key:**
   - Sign up at [NewsAPI.org](https://newsapi.org/) to obtain your API key.

4. **Set up environment variables:**
   - Create a `.env` file in your project directory and add your NewsAPI key:
     ```
     NEWS_API_KEY=your_actual_api_key_here
     ```

## Usage

Run the script with the model of your choice:

```bash
python3 data_fetch.py # fetches the last news
python3 sentiment_analysis.py 1   # Uses 'j-hartmann/emotion-english-distilroberta-base', and saves result as csv. 
python3 sentiment_analysis.py 2   # Uses 'cardiffnlp/twitter-roberta-base-emotion', and saves result as csv. 
```

- `1` selects **j-hartmann/emotion-english-distilroberta-base**
- Any other value selects **cardiffnlp/twitter-roberta-base-emotion**

## Example Output

I am using the output to create bar / pie charts, using PowerBI. 

### sentiment analysis
<img width="1116" height="594" alt="Screenshot 2025-08-01 at 11 19 22 AM" src="https://github.com/user-attachments/assets/08ca64b1-93aa-46cf-b406-94058e6666dc" />

### topic clustering LDA
<img width="943" height="530" alt="Screenshot 2025-08-01 at 4 09 44 PM" src="https://github.com/user-attachments/assets/887e9770-20f4-42d1-90ba-1b0be06c368f" />

### topic clustering NMF
<img width="938" height="520" alt="Screenshot 2025-08-01 at 4 10 02 PM" src="https://github.com/user-attachments/assets/51330f8d-b3b3-4fb9-ac69-e0ba2b5130d5" />


## Requirements

- Python 3.7+
- [transformers](https://pypi.org/project/transformers/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [requests](https://pypi.org/project/requests/)

Install all with:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
transformers
python-dotenv
requests
```

## License

MIT License

---

## Notes

- This project is for educational and research purposes.
- Be mindful of [NewsAPI usage limits](https://newsapi.org/pricing).
- For emotion models, see their respective HuggingFace pages for more details.
