import pandas as pd
import numpy as np
import os

# Define the base path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'MINDsmall_train'))

# Load news data
news = pd.read_csv(
    os.path.join(base_path, 'news.tsv'),
    sep='\t', header=None,
    names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
)

# Load behaviors (user clicks)
behaviors = pd.read_csv(
    os.path.join(base_path, 'behaviors.tsv'),
    sep='\t', header=None,
    names=['impression_id', 'user_id', 'timestamp', 'history', 'impressions']
)

# Load entity embeddings (e.g., for "Apple Inc." or "NBA")
entity_embeddings = {}
with open(os.path.join(base_path, 'entity_embedding.vec'), 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        entity_embeddings[parts[0]] = np.array([float(x) for x in parts[1].split()])

# Debugging: Print the first few records to check if data is loaded correctly
print("News data sample:\n", news.head())
print("\nEntity embeddings for 'Apple':", entity_embeddings.get('Apple', None))

# Function to get article embeddings by averaging title and abstract entity embeddings
def get_article_embeddings(row):
    """Combine title + abstract entity embeddings for an article."""
    entities = []
    if isinstance(row['title_entities'], str):
        entities.extend(row['title_entities'].split(' '))
    if isinstance(row['abstract_entities'], str):
        entities.extend(row['abstract_entities'].split(' '))
    
    # Average embeddings of all entities in the article
    embeddings = [entity_embeddings[e] for e in entities if e in entity_embeddings]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(100)  # Assume 100D embeddings

# Apply the function to get article embeddings and add it as a new column
news['article_embedding'] = news.apply(get_article_embeddings, axis=1)

print("\nSample of news data with article embeddings:\n", news[['news_id', 'article_embedding']].head())
