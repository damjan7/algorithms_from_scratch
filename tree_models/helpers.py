import numpy as np

def create_random_dataset(n_samples, n_features=10):
    np.random.seed(42)

    # Define dataset parameters
    n_samples = n_samples  # Number of observations
    n_features = n_features    # Number of continuous predictors

    # Generate random features from a standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Define true coefficients for linear components
    true_coeffs = np.random.uniform(-2, 2, size=n_features)

    # Generate response variable with nonlinear and linear effects
    noise = np.random.normal(scale=1.0, size=n_samples)
    y = (
        1.5 * X[:, 0] ** 2 +       # Quadratic effect
        2 * np.sin(X[:, 1]) +      # Sinusoidal effect
        X[:, 2] * X[:, 3] +        # Interaction term
        3 * np.log1p(abs(X[:, 4])) +  # Log transformation
        X[:, 5:10] @ true_coeffs[5:10] +  # Some linear contributions
        noise  # Add Gaussian noise
    )
    return X, y, true_coeffs


def decode_sklearn_decision_tree(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=np.around(values[i], 3)
                )
            )
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=np.around(values[i], 3),
                )
            )