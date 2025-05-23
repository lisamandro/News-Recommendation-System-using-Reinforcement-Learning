import numpy as np

class LinUCB:
    def __init__(self, n_features, alpha=0.5):
        self.A = np.eye(n_features)  # Identity matrix
        self.b = np.zeros(n_features)
        self.alpha = alpha  # Exploration factor

    def recommend(self, user_features, articles):
        """Articles: {article_id: embedding}"""
        scores = {}
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        
        for article_id, article_embedding in articles.items():
            x = np.concatenate([user_features, article_embedding])
            score = theta @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
            scores[article_id] = score
        return max(scores, key=scores.get)

    def update(self, user_features, chosen_embedding, reward):
        x = np.concatenate([user_features, chosen_embedding])
        self.A += np.outer(x, x)
        self.b += reward * x