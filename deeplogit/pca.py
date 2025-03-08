import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_principal_components(
    embeddings,
    num_components=3,
    principal_components_path=None,
    save_to_csv=True,
):
    """Compute principal components for multiple embedding matrices and combined embeddings.

    Args:
        embeddings (dict): Dictionary of (J, num_features) matrices where the key is the model.
        combined_embeddings (np.ndarray): The combined feature matrix of shape (J, total_num_features).
        principal_components_path (str): The path to save the principal components.
        num_components (int, optional): Number of principal components to compute. Defaults to 10.

    Returns:
        dict: Dictionary of principal component matrices for each model and combined embeddings.
    """
    # Initialize dictionary to store all principal components
    all_principal_components = {}

    # Compute PCA for each model's embeddings
    for model_name, embedding_matrix in embeddings.items():
        non_zero_cols = (embedding_matrix != 0).sum(axis=0) > 0
        embedding_matrix = embedding_matrix[:, non_zero_cols]
        if embedding_matrix.shape[1] == 0:
            # Skip this model if all columns are zero
            print(f"Skipping {model_name} because all columns are zero.")
            continue

        # 2. Standardize (center and scale to unit variance)
        scaler = StandardScaler()
        embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)

        # Initialize and fit PCA
        pca = PCA(n_components=num_components, svd_solver="full")
        principal_components = pca.fit_transform(embedding_matrix_scaled)

        # Store the principal components
        all_principal_components[model_name] = principal_components

        # Save to CSV
        if save_to_csv and principal_components_path:
            os.makedirs(principal_components_path, exist_ok=True)
            principal_components_df = pd.DataFrame(principal_components)
            output_path = os.path.join(
                principal_components_path, f"{model_name}_principal_components.csv"
            )
            principal_components_df.to_csv(output_path, index=False)

    return all_principal_components
