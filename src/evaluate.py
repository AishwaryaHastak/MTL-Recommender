import numpy as np

import scipy.stats as st


FLOAT_MAX = np.finfo(np.float32).max


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: class:Interactions
        Test interactions.
    train: class:Interactions, optional
        Train interactions. If supplied, scores of known
        interactions will be set to very low values and so not
        affect the MRR.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each user in test.
    """

    test = test.tocsr()
    print(test)
    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):
    # for batch in test:
        # user_id, row = batch['user_id'], batch['item_id']
        print('user_id',user_id, 'row',row)
        if not len(row.indices):
            continue

        predictions, _ = model.predict(user_id)
        predictions *= -1.0

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()
        print('predictions',predictions)
        print('row.indices',row.indices)
        mrrs.append(mrr)

    return np.array(mrrs).mean()


def mse_score(model, test):
    """
    Compute MSE score for test interactions.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: class:Interactions
        Test interactions.

    Returns
    -------

    rmse_score: float
        The MSE score.
    """
    mse_scores = []
    for batch in test:
        user_ids, item_ids,ratings = batch['user_id'], batch['item_id'], batch['rating']
        _, scores = model.predict(user_ids, item_ids)
        mse_scores.append(((ratings - scores)**2).detach().numpy().mean())
    return np.array(mse_scores).mean()
