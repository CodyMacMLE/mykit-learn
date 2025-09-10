import mykitlearn

import numpy as np

def test_LinearRegression():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ], dtype=np.float64)
    y = np.array([5.0, 7.0, 9.0], dtype=np.float64)

    linearModelBatch = mykitlearn.LinearRegression()
    linearModelMiniBatch = mykitlearn.LinearRegression(batches=2)
    linearModelStochastic = mykitlearn.LinearRegression(gd_type="SGd")

    linearModelBatch.train(X, y)
    linearModelMiniBatch.train(X, y)
    linearModelStochastic.train(X, y)

    assert np.allclose(linearModelBatch.predict(X), y, atol=0.5)
    assert np.allclose(linearModelMiniBatch.predict(X), y, atol=0.5)
    assert np.allclose(linearModelStochastic.predict(X), y, atol=0.5)

def test_LogisticRegression():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])
    y = np.array([0.0, 1.0], dtype=np.float64)

    logisticModelBatch = mykitlearn.LogisticRegression()
    """
    logisticModelMiniBatch = mykitlearn.LogisticRegression(batches=2)
    logisticModelStochastic = mykitlearn.LogisticRegression(gd_type="SGD")
    """

    logisticModelBatch.train(X, y)
    predsLogisticBatch = logisticModelBatch.predict(X)
    """
    logisticModelMiniBatch.train(X, y)
    logisticModelStochastic.train(X, y)
    """

    assert np.all((predsLogisticBatch >= 0) & (predsLogisticBatch <= 1))
    """
    assert np.allclose(logisticModelMiniBatch.predict(X), y, atol=0.5)
    assert np.allclose(logisticModelStochastic.predict(X), y, atol=0.5)
    """