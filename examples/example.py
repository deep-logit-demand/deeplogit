import pandas as pd

from deeplogit import DeepLogit


def main():
    input_dir_path = "example_data/"

    # Load data
    long_choice_data = pd.read_csv(input_dir_path + "long_choice_data.csv")
    variables_attributes = [
        "price",
        "position",
    ]

    images_dir_path = input_dir_path + "images/"
    titles_csv_path = input_dir_path + "texts/titles.csv"
    descriptions_csv_path = input_dir_path + "texts/descriptions.csv"
    reviews_csv_path = input_dir_path + "texts/reviews.csv"

    # unstructured_data_list = [
    #     ("reviews", reviews_csv_path),
    #     ("titles", titles_csv_path),
    #     ("descriptions", descriptions_csv_path),
    #     ("images", images_dir_path),
    # ]

    unstructured_data_list = [
        ("reviews", reviews_csv_path),
    ]

    results = {}

    for (
        unstructured_data_name,
        unstructured_data_path,
    ) in unstructured_data_list:
        print(f"Running model with {unstructured_data_name} as unstructured data")
        results[unstructured_data_name] = {}

        model = DeepLogit()

        # Fit the model
        model.fit(
            data=long_choice_data,
            variables=variables_attributes,
            unstructured_data_path=unstructured_data_path,
            select_optimal_PC_RCs=True,
            number_of_PCs=4,
            n_draws=100,
            n_starting_points=100,
            print_results=True,
        )

        # Predict market shares and diversion ratios
        predicted_market_shares = model.predict(data=long_choice_data)
        predicted_diversion_ratios = model.predict_diversion_ratios(data=long_choice_data)

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
