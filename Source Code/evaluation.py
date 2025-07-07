import dataset
import random
import time


def evaluate(policy, size=100, learn_ratio=0.9):
    """
    Evaluates a bandit policy on logged MIND-small interactions.

    Parameters
    ----------
    policy : object
        Bandit algorithm instance (e.g., LinearContextualBandit)
    size : int
        Percentage of dataset to use
    learn_ratio : float
        Fraction of interactions used for learning (vs. deployment)

    Returns
    -------
    learn : list
        CTR progression for learning interactions
    deploy : list
        CTR progression for deployment interactions
    """

    start = time.time()
    learn_clicks, learn_trials = 0, 0
    deploy_clicks, deploy_trials = 0, 1

    learn = []
    deploy = []

    if size == 100:
        events = dataset.events
    else:
        k = int(dataset.n_events * size / 100)
        events = random.sample(dataset.events, k)

    for t, event in enumerate(events):
        displayed = event[0]
        reward = event[1]
        user = event[2]
        pool_idx = event[3]

        chosen = policy.select_arm(learn_clicks + deploy_clicks, user, pool_idx)

        if chosen == displayed:
            if random.random() < learn_ratio:
                learn_clicks += reward
                learn_trials += 1
                policy.update(displayed, reward, user, pool_idx)
                learn.append(learn_clicks / learn_trials)
            else:
                deploy_clicks += reward
                deploy_trials += 1
                deploy.append(deploy_clicks / deploy_trials)

    duration = round(time.time() - start, 1)
    time_display = f"{round(duration / 60, 1)}m" if duration > 60 else f"{duration}s"

    print(f"{policy.algorithm:<25}CTR: {round(deploy_clicks / deploy_trials, 4):<8}Time: {time_display}")
    return learn, deploy
