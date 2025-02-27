import polars as pl
from pathlib import Path


def get_customer_df_benchmarks(data_path: Path, config: dict):
    """Processes customer data with age grouping and zip code mapping.

    Args:
        data_path (Path): Path to directory containing 'customers.csv' and 'zip_code_count.csv'.
        config (dict): Configuration with 'min_zip_code_count'. Updated with 'num_age_groups' and 'num_zip_codes'.

    Returns:
        pl.DataFrame: Processed DataFrame with customer_id, age_group (0-6), and mapped zip_code_id.
    """
    file_path = data_path / "customers.csv"
    df = pl.scan_csv(file_path).select(
        (
            "customer_id",
            pl.col("age").fill_null(strategy="mean"),
            "postal_code",
        )
    )

    # df = df.with_columns(
    #     [
    #         pl.when(pl.col("age").is_null())
    #         .then(0)
    #         .when(pl.col("age") < 25)
    #         .then(1)
    #         .when(pl.col("age").is_between(25, 34))
    #         .then(2)
    #         .when(pl.col("age").is_between(35, 44))
    #         .then(3)
    #         .when(pl.col("age").is_between(45, 54))
    #         .then(4)
    #         .when(pl.col("age").is_between(55, 64))
    #         .then(5)
    #         .otherwise(6)
    #         .alias("age_group")
    #     ]
    # )
    # config["num_age_groups"] = 7

    return df.collect()
