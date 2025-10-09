# deeplogit: Mixed Logit Estimation with Text and Image Embeddings Extracted Using Deep Learning Models

## Overview

This package provides a class [`DeepLogit`](https://github.com/franklinshe/DeepLogit/blob/master/deeplogit/deeplogit.py) that can be used to estimate a mixed logit model with text and image embeddings extracted using deep learning models. The class provides methods to preprocess text and image data, fit the model, and make predictions.

The package estimates models using four machine learning models for images and four machine learning models for texts. For images, the machine learning models used are: [VGG19](https://arxiv.org/abs/1409.1556), [ResNet50](https://arxiv.org/abs/1512.03385), [Xception](https://arxiv.org/abs/1610.02357), and [InceptionV3](https://arxiv.org/abs/1512.00567). For texts, the machine learning models used are: [Count](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer), [USE](https://arxiv.org/abs/1803.11175), and [ST](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). The package will estimate mixed logit models using principal components extracted each machine learning architecture separately and select the best model which achieves the best performance. The package will either apply a selection algorithm to choose the combination of principal components which minimizes AIC or estimate the model with the maximum number of principal components selected. 

Choice attributes can also be included (see the example, where price and position attributes are used). 

The DeepLogit package relies heavily on the xlogit library for the implementation of the mixed logit model. For more information on the xlogit library, see the [xlogit repository](https://github.com/arteagac/xlogit/).


## Installation

This package is supported on Python 3.9. Please make sure you have Python 3.9 installed before proceeding.

This package is available on PyPI [here](https://pypi.org/project/deeplogit/). You can install it using pip:

```bash
pip install deeplogit
```

## Example

### 1. Import libraries and load data

```python
import pandas as pd
from deeplogit import DeepLogit

# Load long choice data
input_dir_path = "example_data/"
long_choice_data_path = input_dir_path + "long_choice_data.csv"

# Define structured attribute names
variables_attributes = ["price", "position"]

# Define unstructured data file paths
descriptions_csv_path = input_dir_path + "texts/descriptions.csv"
images_dir_path = input_dir_path + "images/"
```

### 2. Initialize and fit a model for each unstructured data type

```python
# Initialize the model
model = DeepLogit()

# For this example, let's fit a model using image data
unstructured_data_path = images_dir_path

# Fit the model
model.fit(
    data_path=long_choice_data_path,
    variables=variables_attributes,
    unstructured_data_path=unstructured_data_path,
    select_optimal_PC_RCs=True,
    number_of_PCs=6,
    n_draws=100,
    n_starting_points=100,
    print_results=True,
    limit_cores=True,
)
```

The `fit` method has the following parameters:
- `data` : pandas.DataFrame
    The choice data in long format where each observation is a consumer-product pair. Must contain the following columns:
    - choice_id: Consumer identifier
    - product_id: Product identifier
    - choice: The choice indicator (1 for chosen alternative, 0 otherwise).
    - price: The price of the product.
- `variables` : list
    The list of variable names that vary both across products and consumers. The names must match the column names in the data. Must include the price variable.
- `unstructured_data_path` : str
    The path to the unstructured data. If the data is images, this should be the path to the directory containing the images. If the data is text, this should be the path to the CSV file containing the text data.
- `select_optimal_PC_RCs` : bool, optional
    True to select the AIC-minimizing combination of principal components via brute force algorithm. False to include all principal components without optimization. Default is True.
- `number_of_PCs` : int, optional
    The number of principal components to extract from the unstructured data. Default is 3.
- `n_draws` : int, optional
    The number of draws to approximate mixing distributions of the random coefficients. Default is 100.
- `n_starting_points` : int, optional
    The number of starting points to use in the estimation. Default is 100.
- `print_results` : bool, optional
    Whether to print the results of each model fit. Default is True.
- `limit_cores` : bool, optional
    Whether to limit the number of CPU cores used for estimation. Default is True.

### 3. Access model diagnostics

```python
# Print model diagnostics
print(f"Fitted model log-likelihood: {model.loglikelihood}")
print(f"Fitted model AIC: {model.aic}")
print(f"Fitted model estimate names: {model.coeff_names}")
print(f"Fitted model estimate values: {model.coeff_}")
print(f"Fitted model estimate standard errors: {model.stderr}")
```

### 4. Make predictions

```python
# Predict market shares (J x N matrix)
predicted_market_shares = model.predict()

# Predict diversion ratios (J x J matrix)
predicted_diversion_ratios = model.predict_diversion_ratios()
```

### 5. Another fit example

```python
# Example: Use images, extract 6 PCs, and include all PCs without optimization
model.fit(
    data_path=long_choice_data_path,
    variables=variables_attributes,
    unstructured_data_path=images_dir_path,
    select_optimal_PC_RCs=False,
    number_of_PCs=6,
    n_draws=100,
    n_starting_points=100,
    print_results=True,
    limit_cores=True,
)
```

For a full example, including the dataset used in this example, see the `examples/` directory in the repository.

## Contributors

We are deeply grateful to Franklin She for transforming our existing code into a user-friendly, well-documented Python package. We also thank Celina Park, Tanner Parsons, James Ryan, Janani Sekar, Andrew Sharng, Adam Sy, and Vitalii Tubdenov for their exceptional research assistance. The code they developed in the past either inspired this package or was directly used to build it.

## Citing DeepLogit

Our **DeepLogit** package borrows extensively from the code of the existing **Xlogit** package [(see here)](https://github.com/arteagac/xlogit). Therefore, if you use **DeepLogit** in your work, please cite both our package and the original **Xlogit** package as follows:

Compiani, G., Morozov, I., & Seiler, S. (2023). Demand estimation with text and image data. SSRN Working Paper. [Link to paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4588941)

Arteaga, C., Park, J., Beeramoole, P. B., & Paz, A. (2022). Xlogit: An open-source Python package for GPU-accelerated estimation of Mixed Logit models. Journal of Choice Modelling, 42, 100339. [Link to paper](https://doi.org/10.1016/j.jocm.2021.100339)
