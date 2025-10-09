import pandas as pd

from deeplogit import DeepLogit

# The below provides an example of applying the DeepLogit model across image data and three types of text data: product reviews, titles, and descriptions.

def main():
    input_dir_path = "example_data/"

    # We start by loading the choice data
    # Data must input in long format
    # Each row contains: the decision-maker's unique identifier, the product's unique identifier, the choice indicator (1 if chosen, 0 otherwise), and additional product attributes (in our case product price and position)
    long_choice_data = input_dir_path + "long_choice_data.csv"
    
    # We specify the additional product attributes to be included in the model
    variables_attributes = [
        "price",
        "position",
    ]

    # We then define the paths to the unstructured data
    # Note that identifiers for text and image data must match the product identifiers in the choice data
    images_dir_path = input_dir_path + "images/"
    titles_csv_path = input_dir_path + "texts/titles.csv"
    descriptions_csv_path = input_dir_path + "texts/descriptions.csv"
    reviews_csv_path = input_dir_path + "texts/reviews.csv"

    unstructured_data_list = [
        ("reviews", reviews_csv_path),
        ("titles", titles_csv_path),
        ("descriptions", descriptions_csv_path),
        ("images", images_dir_path),
    ]

    results = {}

    # We iterate over each type of unstructured data
    for (
        unstructured_data_name,
        unstructured_data_path,
    ) in unstructured_data_list:
        print(f"Running model with {unstructured_data_name} as unstructured data")
        results[unstructured_data_name] = {}

        # Initialize the DeepLogit model
        model = DeepLogit()

        # Fit the model
        model.fit(
            data_path=long_choice_data,
            variables=variables_attributes,
            unstructured_data_path=unstructured_data_path,
            select_optimal_PC_RCs=False,
            number_of_PCs=6,
            n_draws=100,
            n_starting_points=100,
            print_results=True,
            limit_cores=True,
        )

        # Predict market shares and diversion ratios
        predicted_market_shares = model.predict()
        predicted_diversion_ratios = model.predict_diversion_ratios()

        # Save selected embedding model and specification
        results[unstructured_data_name][
            "selected_embedding_model"
        ] = model.best_embedding_model
        results[unstructured_data_name][
            "selected_specification"
        ] = model.best_specification

        # Save model fit statistics
        results[unstructured_data_name]["log_likelihood"] = model.loglikelihood
        results[unstructured_data_name]["aic"] = model.aic

        # Save estimated parameters and standard errors
        results[unstructured_data_name]["coefficient_names"] = model.coeff_names
        results[unstructured_data_name]["estimated_coefficients"] = model.coeff_
        results[unstructured_data_name]["standard_errors"] = model.stderr

        # Save predicted market shares and diversion ratios
        results[unstructured_data_name][
            "predicted_market_shares"
        ] = predicted_market_shares
        results[unstructured_data_name][
            "predicted_diversion_ratios"
        ] = predicted_diversion_ratios

    print("\n\nResults:")
    print("==" * 40)

    for unstructured_data_name, result in results.items():
        print(f"\nResults for {unstructured_data_name} as unstructured data")
        print("Selected embedding model:", result["selected_embedding_model"])
        print("Selected specification:", result["selected_specification"])
        print("Log-likelihood:", result["log_likelihood"])
        print("AIC:", result["aic"])
        print("Coefficient names:")
        print(result["coefficient_names"])
        print("Estimated coefficients:")
        print(result["estimated_coefficients"])
        print("Standard errors:")
        print(result["standard_errors"])
        print("\n")

    print("==" * 40)

    for unstructured_data_name, result in results.items():
        print(f"\nResults for {unstructured_data_name} as unstructured data")
        # print("Predicted market shares:")
        # print(result["predicted_market_shares"])
        print("Predicted diversion ratios:")
        print(result["predicted_diversion_ratios"])
        print("\n")

    print("==" * 40)


if __name__ == "__main__":
    main()
