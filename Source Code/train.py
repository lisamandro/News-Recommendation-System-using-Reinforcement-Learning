from load_data import news, behaviors
from linucb import LinUCB
import numpy as np

# Normalize news ID to have 'N' prefix (to match impressions)
def normalize_news_id(nid):
    if not nid.startswith('N'):
        return 'N' + nid
    return nid

# Create a dict of news_id -> embedding with normalized news IDs
news_embedding_dict = {
    normalize_news_id(row['news_id']): row['article_embedding']
    for _, row in news.iterrows()
}

# Initialize LinUCB model: 100 user features + 100 article embeddings = 200 features
model = LinUCB(n_features=200, alpha=0.5)

correct = 0
total = 0

for _, row in behaviors.head(1000).iterrows():
    user_feats = np.zeros(100)  # Dummy user features (replace with real data)

    # Extract candidate article IDs from impressions, normalize IDs
    candidate_ids = [imp.split('-')[0] for imp in row['impressions'].split()]
    candidate_ids = [normalize_news_id(cid) for cid in candidate_ids]

    # Filter news articles that are candidates (and have embeddings)
    candidates = [cid for cid in candidate_ids if cid in news_embedding_dict]

    if not candidates:
        # No valid candidates; skip this row
        continue

    # Prepare dict for LinUCB recommend input
    article_dict = {cid: news_embedding_dict[cid] for cid in candidates}

    # Recommend article with highest LinUCB score
    chosen = model.recommend(user_feats, article_dict)

    # Get clicked news (those impressions with '-1' means clicked)
    clicked_news = [imp.split('-')[0] for imp in row['impressions'].split() if imp.endswith('-1')]
    clicked_news = [normalize_news_id(cid) for cid in clicked_news]

    # Reward = 1 if chosen article is clicked, else 0
    reward = 1 if chosen in clicked_news else 0

    # Update LinUCB model with feedback
    model.update(user_feats, article_dict[chosen], reward)

    # Track performance
    correct += reward
    total += 1

if total > 0:
    print(f"CTR: {correct / total:.2%}")
else:
    print("No valid candidate articles found in behaviors data.")
