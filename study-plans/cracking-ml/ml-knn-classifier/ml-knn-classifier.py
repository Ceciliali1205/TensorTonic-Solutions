import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    predictions = []
    for point in X_test:
        # sorted takes each index and passes into the lambda
        order = sorted(range(len(X_train)), key=lambda index:(float(np.sum((X_train[index] - point) ** 2)), index))
        counts = {}
        for index in order[:k]:
            label = y_train[index]
            # look up label in the dictionary. If it doesn't exist, return 0.
            counts[label] = counts.get(label, 0) + 1
        # iterate over dictionary keys
        # he key function only tells min() how to compare elements, not what to return
        predictions.append(int(min(counts, key=lambda label: (-counts[label], label))))
    return predictions
