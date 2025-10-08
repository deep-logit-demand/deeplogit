import os
from functools import partial
from glob import glob
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from xlogit import MixedLogit

from .embeddings import generate_image_embeddings, generate_text_embeddings
from .pca import compute_principal_components

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TFHUB_CACHE_DIR"] = os.path.join(
    os.path.expanduser("~"),
    ".deeplogit",
    "tfhub_modules",
)


class DeepLogit:
    """Class for estimation of mixed logit models with image or text embeddings.

    Attributes:
        model : xlogit.MixedLogit
            The fitted mixed logit model.

        best_specification : str
            The best specification of the model.

        best_embedding_model : str
            The best embedding model used in the model.

        best_varnames : list
            The list of variable names used in the model.

    Attributes (inherited from xlogit.MixedLogit):
        ceoff_ : numpy.ndarray
            Estimated coefficients of the model.

        coeff_names : numpy.ndarray
            Names of the estimated coefficients.

        stderr : numpy.ndarray
            Standard errors of the estimated coefficients.

        zvalues : numpy.ndarray
            Z-values of the estimated coefficients.

        pvalues : numpy.ndarray
            P-values of the estimated coefficients.

        loglikelihood : float
            Log-likelihood of the model at the end of estimation.

        convergence : bool
            Whether the model has converged during estimation.

        total_iter : int
            Total number of iterations executed during estimation

        sample_size : int
            Number of samples used in estimation.

        aic : float
            Akaike Information Criterion of the estimated model.

        bic : float
            Bayesian Information Criterion of the estimated model.
    """

    def __init__(self):
        """Initializes the DeepLogit object."""
        self.model = None
        self.best_specification = None
        self.best_embedding_model = None
        self.best_varnames = None
        self.variables = None
        self.unstructured_data_path = None
        self.number_of_PCs = None

    def fit(
        self,
        data,
        variables,
        unstructured_data_path,
        select_optimal_PC_RCs=True,
        number_of_PCs=6,
        n_draws=100,
        n_starting_points=100,
        print_results=True,
    ):
        """Fits a mixed logit model with unstructured data embeddings.

        Args:
            data : pandas.DataFrame
                The choice data in long format where each observation is a consumer-product pair. Must contain the following columns:
                - choice_id: Consumer identifier
                - product_id: Product identifier
                - choice: The choice indicator (1 for chosen alternative, 0 otherwise).
                - price: The price of the product.

            variables : list
                The list of variable names that vary both across products and consumers. The names must match the column names in the data. Must include the price variable.

            unstructured_data_path : str
                The path to the unstructured data. If the data is images, this should be the path to the directory containing the images. If the data is text, this should be the path to the CSV file containing the text data.

            select_optimal_PC_RCs : bool, optional
                True to select the AIC-minimizing combination of principal components via selection algorithm described in "Demand Estimation with text and image data". 
                False to include all principal components without optimization. Default is True.

            number_of_PCs : int, optional
                The number of principal components to extract from the unstructured data. Default is 3.

            n_draws : int, optional
                The number of draws to approximate mixing distributions of the random coefficients. Default is 100.

            n_starting_points : int, optional
                The number of starting points to use in the estimation. Default is 100.

            print_results : bool, optional
                Whether to print the results of each model fit. Default is True.

        Returns:
            None
        """

        data, principal_components_matrices = self._reshape_data(choice_data=data, 
                                                                 unstructured_data_path=unstructured_data_path, 
                                                                 variables=variables, 
                                                                 number_of_PCs=number_of_PCs
                                                                )
        
        self.unstructured_data_path = unstructured_data_path
        self.variables = variables
        self.number_of_PCs = number_of_PCs
        self.unique_products = data["product_id"].unique()

        # 4. Fit mixed logit models and select the best one
        best_model = None
        best_specification = None
        best_embedding_model = None
        best_varnames = None
        best_AIC = np.inf

        for model_name in principal_components_matrices.keys():
            varnames = variables + [
                f"{model_name}_pc{i}" for i in range(1, number_of_PCs + 1)
            ]

            pc_specifications = self._generate_pc_specifications(
                number_of_PCs, model_name
            )

            # Select the model with most random coefficients if not optimizing
            if not select_optimal_PC_RCs:
                max_randvars = max(
                    [len(randvars) for randvars in pc_specifications.values()]
                )
                pc_specifications = {
                    k: v for k, v in pc_specifications.items() if len(v) == max_randvars
                }

                for specification, randvars in pc_specifications_step_j.items():
                        if print_results:
                            print("-" * 50)
                            print(f"Trying model: {model_name}, specification: {specification}")
                        model = self._estimate_mixed_logit(
                            data=data,
                            varnames=varnames,
                            randvars=randvars,
                            n_draws=n_draws,
                            num_starting_points=n_starting_points,
                        )
                        if print_results:
                            print(f"LL: {model.loglikelihood}, AIC: {model.aic}")
                        if model.aic < best_AIC:
                            best_model = model
                            best_AIC = model.aic
                            best_specification = specification
                            best_embedding_model = model_name
                            best_varnames = varnames
            # Otherwise apply selection algorithm
            else:
                best_AIC_model = np.inf
                best_specification_model = None

                for j in range(0, number_of_PCs + 2):
                    if print_results:
                        print("=" * 50)
                        print(f"Estimating for specifications with {j} random coefficients for model {model_name}")

                    pc_specifications_step_j = {
                        specification: randvars
                        for specification, randvars in pc_specifications.items()
                        if specification.count("PC") + specification.count("price") == j
                    }
                    
                    aic_improved = False

                    for specification, randvars in pc_specifications_step_j.items():
                        if print_results:
                            print("-" * 50)
                            print(f"Trying model: {model_name}, specification: {specification}")
                        model = self._estimate_mixed_logit(
                            data=data,
                            varnames=varnames,
                            randvars=randvars,
                            n_draws=n_draws,
                            num_starting_points=n_starting_points,
                        )
                        if print_results:
                            print(f"LL: {model.loglikelihood}, AIC: {model.aic}")
                        if model.aic < best_AIC:
                            best_model = model
                            best_AIC = model.aic
                            best_specification = specification
                            best_embedding_model = model_name
                            best_varnames = varnames
                        if model.aic < best_AIC_model:
                            best_AIC_model = model.aic
                            best_specification_model = specification
                            aic_improved = True
                    if not aic_improved:
                        if print_results:
                            print("=" * 50)
                            print(f"No AIC improvement for models with {j} random coefficients over models with {j-1} random coefficients. Stopping search.")
                            print(f"Best random coefficient specification for {model_name}: {best_specification_model} with AIC: {best_AIC_model}")
                        break

        # 5. Store the best model and the constructed data dataframe and print if required
        self.model = best_model
        self.best_specification = best_specification
        self.best_embedding_model = best_embedding_model
        self.best_varnames = best_varnames

        if print_results:
            print("\n" + "=" * 50)
            print("Best model summary:")
            print(f"Embedding model: {self.best_embedding_model}")
            print(f"Specification: {self.best_specification}")
            print(f"AIC: {self.model.aic}")

    def predict(self, data, seed=1, avail=None):
        """Predicts the choice probabilities for the given data using the fitted model.

        Args:
            data : pandas.DataFrame
                The choice data in long format. Must contain the following columns:
                - choice_id: The ID of the choice situation.
                - product_id: The ID of the product.

            seed : int, optional
                The random seed to use for the prediction. Default is 1.

            avail: numpy.ndarray, optional
                A binary array indicating the availability of each product in each choice situation.
                If None, all products are assumed to be available. Default is None.

        Returns:
            numpy.ndarray: The predicted choice probabilities.
        """
        assert self.model is not None, "Model has not been fitted yet."

        data, _ = self._reshape_data(choice_data=data, 
                                     unstructured_data_path=self.unstructured_data_path, 
                                     variables=self.variables, 
                                     number_of_PCs=self.number_of_PCs
                                     )
        
        unique_products = data["product_id"].unique()

        assert unique_products.all() == self.unique_products.all(), "Product IDs in the data do not match those in the fitted model."

        _, predicted_probs = self.model.predict(
            X=data[self.best_varnames],
            varnames=self.best_varnames,
            ids=data["choice_id"],
            alts=data["product_id"],
            avail=avail,
            return_proba=True,
            halton=False,
            random_state=seed,
        )

        return predicted_probs

    def predict_diversion_ratios(self, data):
        """Predicts the diversion ratios for the given data using the fitted model.

        Args:
        data : pandas.DataFrame the choice data in long format. Must contain the following columns:
            - choice_id: The ID of the choice situation.
            - product_id: The ID of the product. The product ids must be the same as in the data used to fit the model.
            - choice: The choice indicator (1 for chosen alternative, 0 otherwise).

        Returns:
            numpy.ndarray: The predicted diversion ratios.
        """
        assert self.model is not None, "Model has not been fitted yet."

        # Extract first and second choice indices
        unique_products = data["product_id"].unique()

        assert unique_products.all() == self.unique_products.all(), "Product IDs in the data do not match those in the fitted model."

        J = len(unique_products)

        # Prepare an empty count matrix: (J x J).
        # predicted_diversion_matrix[j1, j2] = probability j2 is choice in limited choice set (i.e. with first choice removed)
        # given that product j1 was chosen first.
        predicted_diversion_matrix = np.zeros((J, J), dtype=float)
        for j, product in enumerate(unique_products):

            # remove product j from available choices
            product_j_removed = data.apply(
                lambda row: 0 if row["product_id"] == product else 1,
                axis=1,
            )
            #compute predicted probabilities for all individuals in the full choice set
            s_unconditional = self.predict(data=data)

            #compute predicted new probabilities for all individuals after removal of product j
            s_with_j_removed = self.predict(
                data=data,
                avail=product_j_removed
            )          

            # formula for computing s_j->k = average of (s_k^(j) - s_k) / s_j over all individuals
            # where s_k^(j) is the predicted probability of product k when product j is removed from the choice set
            # s_k is the predicted probability of product k in the full choice set
            # s_j is the predicted probability of product j in the full choice set
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = s_with_j_removed[:, :] - s_unconditional[:, :]
                denom = s_unconditional[:, j]
                probs = diff / denom[:, None]
            
            # set probabilities for the first choice product to 0
            probs[:, j] = 0
            # average across all individuals who chose product j first
            avg_probs = np.average(probs, axis=0)

            predicted_diversion_matrix[j, :] = avg_probs

        assert np.allclose(predicted_diversion_matrix.sum(axis=1), 1)

        return predicted_diversion_matrix
    
    def _reshape_data(self, choice_data, unstructured_data_path, variables, number_of_PCs):
        """Reshape the data to match structure expected by fit function.

        Args:
            choice_data : pandas.DataFrame the choice data in long format. Must contain the following columns:
                - choice_id: The ID of the choice situation.
                - product_id: The ID of the product. The product ids must be the same as in the data used to fit the model.
                - choice: The choice indicator (1 for chosen alternative, 0 otherwise).
            unstructured_data_path : str
                The path to the unstructured data. 
                If the data is images, this should be the path to the directory containing the images. 
                If the data is text, this should be the path to the CSV file containing the text data.
            variables : list
                The list of variable names that vary both across products and consumers. 
                The names must match the column names in the data. Must include the price variable.
            number_of_PCs : int
                The number of principal components to extract from the unstructured data.

        Returns:
            pandas.DataFrame, dict: The reshaped data and the principal components matrices.
        """
        # Copy the choice data to avoid modifying the original data
        data = choice_data.copy()

        # Create dummy variables for product_id column and add to variables
        product_dummies = pd.get_dummies(
            data["product_id"], prefix="product_id"
        ).astype(int)

        data = pd.concat([data, product_dummies], axis=1)

        # Append the names of the first J - 1 product_id_{product_id} columns to the list of strings variables
        dummy_columns = [
            col for col in product_dummies.columns if col != product_dummies.columns[-1]
        ]
        variables.extend(dummy_columns)

        # Determine the type of unstructured data
        if os.path.isdir(unstructured_data_path):
            unstructured_data_type = "images"
        elif os.path.isfile(unstructured_data_path) and unstructured_data_path.endswith(
            ".csv"
        ):
            unstructured_data_type = "text"
        else:
            raise ValueError(
                "Unstructured data path must be a directory (for images) or a CSV file (for text)"
            )

        # 1. Transform unstructured data into embeddings
        if unstructured_data_type == "images":
            unstructured_data = self._load_images(unstructured_data_path)
            embeddings = generate_image_embeddings(unstructured_data)
        elif unstructured_data_type == "text":
            unstructured_data = self._load_texts(unstructured_data_path)
            embeddings = generate_text_embeddings(unstructured_data)
            text_name = list(embeddings.keys())[0]
            embeddings = embeddings[list(embeddings.keys())[0]]
            embeddings = {f"{text_name}_{k}": v for k, v in embeddings.items()}
        else:
            raise ValueError("Unstructured data type must be 'images' or 'text'")

        # 2. Perform PCA on embeddings
        principal_components_matrices = compute_principal_components(
            embeddings,
            num_components=number_of_PCs,
        )

        # 3. Join principal components with choice data
        principal_components = {}

        for (
            model_name,
            principal_components_matrix,
        ) in principal_components_matrices.items():
            principal_components_df = pd.DataFrame(principal_components_matrix)
            principal_components_df["product_id"] = list(unstructured_data.keys())

            for i, col in enumerate(principal_components_df.columns):
                if col == "product_id":
                    continue
                # NOTE: Normalize each principal component for better convergence
                pc_normalized = self._standardize(principal_components_df[col])
                principal_components[f"{model_name}_pc{i+1}"] = dict(
                    zip(principal_components_df["product_id"], pc_normalized)
                )

        for key, pc_dict in principal_components.items():
            data[key] = data["product_id"].map(lambda x: pc_dict.get(str(x)))
        return data, principal_components_matrices

    def __getattr__(self, name):
        """Override the __getattr__ method to allow access to the attributes of the xlogit.MixedLogit object."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        if name in self.__dict__:
            return self.__dict__[name]

        if self.model is not None:
            if hasattr(self.model.__class__, name) or name in self.model.__dict__:
                return getattr(self.model, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _load_images(self, dir_path):
        images = {}
        image_paths = glob(os.path.join(dir_path, "*.jpg"))
        target_size = (224, 224)

        for path in image_paths:
            path = path.replace("\\", "/")
            asin = os.path.splitext(os.path.basename(path))[0]
            try:
                img = image.load_img(path, target_size=target_size)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                images[asin] = img_array
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        images = dict(sorted(images.items()))

        return images

    def _load_texts(self, file_path):
        texts = {}

        # Load the CSV file
        texts_df = pd.read_csv(file_path)

        # Ensure the first column is 'id'
        columns = texts_df.columns
        assert len(columns) >= 2, f"Expected at least 2 columns, got {len(columns)}"
        assert columns[0] == "id", f"Expected first column to be 'id', got {columns[0]}"

        text_columns = columns[1:]  # Exclude the 'id' column

        for _, row in texts_df.iterrows():
            id = str(row["id"])
            text_dict = {col: str(row[col]) for col in text_columns}
            texts[id] = text_dict

        return texts

    def _standardize(self, series):
        """Standardizes a pandas Series."""
        return (series - series.mean()) / series.std()

    def _simulate_individual(
        self,
        first_choice_data,
        coeff_dict,
        i,
        seed=1,
        include_gumbels=True,
    ):
        """Simulates the utilities for an individual given the first choice data and coefficients."""
        unique_products = first_choice_data["product_id"].unique()
        J = len(unique_products)

        np.random.seed(seed)
        # Extract the subset of rows corresponding to this individual's choice situation
        df_i = first_choice_data.loc[first_choice_data["choice_id"] == i]

        # Draw individual-specific random coefficients (if any)
        individual_coeffs = {}
        for param in coeff_dict.keys():
            # Skip the 'sd_' parameters themselves; these are used for the standard deviation
            # of the random parameters
            if param.startswith("sd."):
                param_mean = param[3:]
                mean_ = coeff_dict[param_mean]
                sd_ = np.abs(coeff_dict[param])
                individual_coeffs[param_mean] = np.random.normal(mean_, sd_)
            else:
                # Otherwise, it's a fixed parameter
                individual_coeffs[param] = coeff_dict[param]

        # Compute utilities for each product
        # Add Gumbel random error for each product as well
        utilities = []
        if include_gumbels:
            epsilons = np.random.gumbel(loc=0, scale=1, size=J)
        else:
            epsilons = np.zeros(J)

        # Sort df_i by product_id to ensure consistent ordering
        df_i = df_i.sort_values(by="product_id")
        df_i_indexed = df_i.set_index("product_id")

        # For each product in unique_products (which are sorted), compute the utility
        for j, j_id in enumerate(unique_products):
            row_ij = df_i_indexed.loc[j_id]

            util_ij = 0.0
            # Sum up the utility from each variable in varnames
            for v in individual_coeffs.keys():
                if v.startswith("sd."):
                    continue
                if v in df_i.columns:
                    util_ij += individual_coeffs[v] * row_ij[v]
                else:
                    raise ValueError(f"Variable {v} not found in df_i.columns")

            # Add the random Gumbel error
            util_ij += epsilons[j]

            utilities.append(util_ij)

        utilities = np.array(utilities)

        return utilities

    def _generate_pc_specifications(self, K, embedding_model):
        """Generates the pc_specifications dictionary based on the number of principal components (K)."""
        base_attributes = ["price"] + [
            f"{embedding_model}_pc{i}" for i in range(1, K + 1)
        ]

        pc_specifications = {}
        pc_specifications["plain logit"] = {}

        # Generate all combinations
        for r in range(1, len(base_attributes) + 1):
            for combo in combinations(base_attributes, r):
                combo_name = []
                for attr in combo:
                    if attr == "price":
                        combo_name.append("price")
                    else:
                        combo_name.append(
                            attr.split("_")[-1].upper()
                        )  # Extract PC1, PC2, etc.
                combo_name = ", ".join(combo_name)

                combo_dict = {attr: "n" for attr in combo}
                pc_specifications[combo_name] = combo_dict

        return pc_specifications

    def _estimate_mixed_logit(
        self,
        data,
        varnames,
        randvars,
        n_draws=100,
        num_starting_points=100,
        seed=1,
        halton=False,
    ):
        """Estimates a mixed logit model with given data and specifications using the xlogit library."""

        fit_func = partial(
            self._fit_single_model,
            data=data,
            varnames=varnames,
            randvars=randvars,
            n_draws=n_draws,
            halton=halton,
        )

        n_cores = max(1, cpu_count() - 1)

        with Pool(n_cores) as pool:
            results = pool.map(
                fit_func,
                range(
                    1 + num_starting_points * seed,
                    num_starting_points + 1 + num_starting_points * seed,
                ),
            )

        # Filter out failed fits and find best model
        valid_results = [(model, ll) for model, ll in results if model is not None]

        if not valid_results:
            raise RuntimeError(
                "All model fits failed. Please check your data and parameters."
            )

        # Get model with highest log-likelihood
        best_model, _ = max(valid_results, key=lambda x: x[1])

        return best_model

    def _fit_single_model(
        self, random_state, data, varnames, randvars, n_draws, halton=False
    ):
        """Helper function to fit a single model with given random state"""
        try:
            model = MixedLogit()
            model.fit(
                X=data[varnames],
                y=data["choice"],
                varnames=varnames,
                ids=data["choice_id"],
                alts=data["product_id"],
                n_draws=n_draws,
                random_state=random_state,
                randvars=randvars,
                halton=halton,
                verbose=0,
            )
            return model, model.loglikelihood
        except Exception as e:
            print(f"Warning: Fitting failed for random_state {random_state}: {str(e)}")
            return None, -np.inf
