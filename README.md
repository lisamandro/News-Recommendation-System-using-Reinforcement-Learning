# News Recommendation System using Reinforcement Learning

This project implements a personalized news recommendation system using the LinUCB algorithm, which balances exploration and exploitation to recommend relevant news articles based on user feedback. The system includes a Streamlit-based interactive web interface where users can receive article recommendations, provide click feedback, and track the click-through rate (CTR).

## Features

- **LinUCB Algorithm**: Contextual bandit approach for personalized recommendations.
- **User Feedback Loop**: Accepts user clicks on recommended articles and updates the model online.
- **CTR Tracking**: Calculates and displays click-through rate based on real and dummy user interactions.
- **Streamlit Interface**: Easy-to-use web UI for interaction and visualization.

## Dataset

- The project uses the MIND dataset, which contains millions of news impressions and user click behaviors.
- Due to size and licensing, the dataset is **not included** in the repository.
- Users should download the dataset from the official [MIND dataset site](https://msnews.github.io/) and place it in the designated folder before running the code.



