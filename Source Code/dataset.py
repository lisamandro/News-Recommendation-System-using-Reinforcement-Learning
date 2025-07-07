import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_mind_dataset(news_path="MINDsmall_train/news.tsv", behavior_path="MINDsmall_train/behaviors.tsv", max_articles=None):
    
    """
    Loads and processes the MINDsmall dataset to extract article features and user interactions.

    Parameters
    ----------
    news_path : str
        Path to the news.tsv file
    behavior_path : str
        Path to the behaviors.tsv file
    max_articles : int or None
        Limit on the number of articles to load

    Populates
    ---------
    articles : list of str
        Article IDs
    features : np.ndarray
        TF-IDF vectorized article titles
    events : list of lists
        Interaction events formatted as [clicked_index, reward, user_vector, pool_indices]
    n_arms : int
        Number of articles
    n_events : int
        Number of usable interactions
    """
    
    global articles, features, events, n_arms, n_events

    news = pd.read_csv(news_path, sep="\t", header=None,
                       names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    news["title"] = news["title"].fillna("")

    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(news["title"])
    features = tfidf_matrix.toarray()

    article2idx = {aid: idx for idx, aid in enumerate(news["id"])}
    idx2article = {v: k for k, v in article2idx.items()}
    articles = list(article2idx.keys())

    if max_articles is not None:
        articles = articles[:max_articles]
        features = features[:max_articles]
        article2idx = {aid: idx for idx, aid in enumerate(articles)}

    behaviors = pd.read_csv(behavior_path, sep="\t", header=None,
                             names=["ImpressionID", "UserID", "Time", "History", "Impressions"])

    events = []
    skipped = 0

    for _, row in behaviors.iterrows():
        try:
            clicked_articles = str(row["History"]).split()
            impressions = str(row["Impressions"]).split()
            pool_ids = [imp.split("-")[0] for imp in impressions]
            labels = [int(imp.split("-")[1]) for imp in impressions]

            if not any([aid in article2idx for aid in pool_ids]):
                continue

            valid_clicks = [aid for aid in clicked_articles if aid in article2idx]
            if valid_clicks:
                user_vec = np.mean([features[article2idx[aid]] for aid in valid_clicks], axis=0)
            else:
                user_vec = np.zeros(features.shape[1])

            pool_idx = []
            clicked_index = None
            for i, aid in enumerate(pool_ids):
                if aid not in article2idx:
                    continue
                pool_idx.append(article2idx[aid])
                if labels[i] == 1:
                    clicked_index = len(pool_idx) - 1

            if pool_idx and clicked_index is not None:
                events.append([clicked_index, 1, user_vec, pool_idx])
            elif pool_idx:
                random_index = np.random.randint(len(pool_idx))
                events.append([random_index, 0, user_vec, pool_idx])
        except:
            skipped += 1

    features = np.array(features)
    n_arms = len(articles)
    n_events = len(events)

    print(f"Loaded {n_events} events with {n_arms} articles. Skipped {skipped} bad rows.")
