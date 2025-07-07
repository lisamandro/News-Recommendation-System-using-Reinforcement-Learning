import sys
sys.path.append('./Source Code')

import dataset
from algorithms import LinUCB, EpsilonGreedy, ThompsonSampling, UCB
from evaluation import evaluate

import matplotlib.pyplot as plt
import numpy as np

# Function to plot CTR
def plot_ctr(ctr_list, label):
    if len(ctr_list) == 0:
        return
    smoothed = np.convolve(ctr_list, np.ones(100)/100, mode='valid')
    plt.plot(smoothed, label=label)

# List of algorithm classes only, don't initialize yet
algorithm_classes = [
    ("LinUCB", lambda: LinUCB(alpha=0.1, context="user")),
    ("EpsilonGreedy", lambda: EpsilonGreedy(epsilon=0.1)),
    ("Thompson", lambda: ThompsonSampling()),
    ("UCB1", lambda: UCB
(alpha=0.1)),
]

# Loop over algorithms
for name, algo_fn in algorithm_classes:
    # Load dataset FIRST before model init
    dataset.load_mind_dataset(
        news_path="Dataset/MINDsmall_train/news.tsv",
        behavior_path="Dataset/MINDsmall_train/behaviors.tsv",
        max_articles=500
    )
    
    model = algo_fn()  # Now dataset.features is available
    learn, deploy = evaluate(model)
    plot_ctr(deploy, model.algorithm)

plt.title("Deployment CTR Over Time")
plt.xlabel("Rounds")
plt.ylabel("CTR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
