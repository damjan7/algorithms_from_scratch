import numpy as np

def generate_classification_dataset(n_samples=10000, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Generate linear features (simple linear relationships)
    linear_features = np.random.randn(n_samples, 5)
    linear_coefs = np.array([1.5, -2.0, 3.0, -1.0, 2.5])
    linear_part = linear_features @ linear_coefs

    # Generate nonlinear features (complex relationships)
    nonlinear_features = np.random.randn(n_samples, 5)
    nonlinear_part = (
        np.sin(nonlinear_features[:, 0]) +
        nonlinear_features[:, 1] ** 2 -
        np.log(np.abs(nonlinear_features[:, 2]) + 1) +
        np.exp(-nonlinear_features[:, 3] ** 2) +
        nonlinear_features[:, 4] * nonlinear_features[:, 1]
    )

    # Combine linear and nonlinear parts, add noise
    combined = linear_part + nonlinear_part + np.random.normal(scale=0.5, size=n_samples)

    # Generate binary response using sigmoid function
    probabilities = 1 / (1 + np.exp(-combined))
    response = (probabilities > 0.5).astype(int)

    # Stack all features
    X = np.hstack([linear_features, nonlinear_features])
    y = response

    return X, y
