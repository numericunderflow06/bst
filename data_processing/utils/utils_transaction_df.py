from datetime import datetime
import json
from pathlib import Path
import polars as pl
from data_processing.utils.stateless_rng import global_rng

def filter_purchases_purchases_per_month_pl(
    df_pl: pl.DataFrame, train_end: datetime.date, group_by_channel_id: bool = False
):
    """Filters extreme customers and groups purchases by date and optionally by sales channel.

    This function:
    1. Groups transactions by customer, date, and optionally sales channel
    2. Identifies extreme customers based on the 99th percentile of total items purchased
    3. Removes these customers from the dataset

    Args:
        df_pl (pl.DataFrame): Input transaction dataframe containing:
            - customer_id: Customer identifier
            - date: Transaction date
            - article_id: Product identifier
            - price: Transaction price
            - sales_channel_id: Sales channel identifier
        train_end (datetime.date): End date for training period.
        group_by_channel_id (bool, optional): Whether to group transactions by sales channel. Defaults to False.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing:
            - grouped_df: Grouped transaction data with columns:
                - customer_id, date, [sales_channel_id], article_ids, total_price, prices, num_items
            - extreme_customers: DataFrame of customers identified as outliers based on purchase behavior

    Notes:
        Extreme customers are identified using the 99th percentile of total items purchased
        during the training period.
    """
    # Used for multi variate time series
    if group_by_channel_id:
        grouped_df = (
            df_pl.lazy()
            .group_by(["customer_id", "date", "sales_channel_id"])
            .agg(
                [
                    pl.col("article_id").explode().alias("article_ids"),
                    pl.col("price").sum().round(2).alias("total_price"),
                    pl.col("price").explode().alias("prices"),
                ]
            )
            .with_columns(pl.col("article_ids").list.len().alias("num_items"))
        )
    else:
        grouped_df = (
            df_pl.lazy()
            .group_by(["customer_id", "date"])
            .agg(
                [
                    pl.col("article_id").explode().alias("article_ids"),
                    pl.col("price").sum().round(2).alias("total_price"),
                    pl.col("sales_channel_id").explode().alias("sales_channel_ids"),
                    pl.col("price").explode().alias("prices"),
                ]
            )
            .with_columns(pl.col("article_ids").list.len().alias("num_items"))
        )

    # Only remove customers with extreme purchases in train period
    customers_summary = (
        df_pl.lazy()
        .filter(pl.col("date") < train_end)
        .group_by("customer_id")
        .agg(
            [
                pl.col("date").n_unique().alias("total_purchases"),
                pl.col("price").sum().round(2).alias("total_spent"),
                pl.col("article_id").flatten().alias("flattened_ids")
            ]
        )
        .with_columns(pl.col("flattened_ids").list.len().alias("total_items"))
    )

    quantile = 0.99
    total_purchases_99, total_spending_99, total_items_99 = (
        customers_summary.select(
            [
                pl.col("total_purchases").quantile(quantile),
                pl.col("total_spent").quantile(quantile),
                pl.col("total_items").quantile(quantile),
            ]
        )
        .collect()
        .to_numpy()
        .flatten()
    )

    # Currently only remove customers with very large number of total items purchased
    extreme_customers = customers_summary.filter(
        (pl.col("total_items") >= total_items_99)
        # | (pl.col("total_purchases") >= total_purchases_99)
        # | (pl.col("total_spent") >= total_spending_99)
    )

    extreme_customers = extreme_customers.select("customer_id").unique()
    extreme_customers = extreme_customers.collect()

    print(
        f"""
        Cutoff Values for {quantile*100}th Percentiles:
        -----------------------------------
        Total items bought:    {total_items_99:.0f} items

        -----------------------------------
        Removed Customers:     {len(extreme_customers):,}
        """
    )

    return grouped_df.collect(), extreme_customers

def train_test_split(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    subset: int = None,
    train_subsample_percentage: float = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits data into train, validation, and test sets with optional subsampling.

    The function performs the following operations:
    1. Optional subsampling of both train and test data
    2. Optional percentage-based subsampling of training data
    3. Creates a validation set from 10% of the training data

    Args:
        train_df (pl.DataFrame): Training dataset.
        test_df (pl.DataFrame): Test dataset.
        subset (int, optional): If provided, limits both train and test sets to first n rows. 
            Defaults to None.
        train_subsample_percentage (float, optional): If provided, randomly samples this percentage 
            of training data. Defaults to None.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Tuple containing:
            - train_df: Final training dataset (90% of training data after subsampling)
            - val_df: Validation dataset (10% of training data)
            - test_df: Test dataset (potentially subsampled)

    Notes:
        If both subset and train_subsample_percentage are provided, subset is applied first.
        The validation set is always 10% of the remaining training data after any subsampling.
    """

    if subset is not None:
        train_df = train_df[:subset]
        test_df = test_df[:subset]
    elif train_subsample_percentage is not None:
        sampled_indices = global_rng.choice(
            len(train_df),
            size=int(train_subsample_percentage * len(train_df)),
            replace=False,
        )
        train_df = train_df[sampled_indices]

    # Train-val-split
    # Calculate 10% of the length of the array
    sampled_indices = global_rng.choice(
        len(train_df), size=int(0.1 * len(train_df)), replace=False
    )
    val_df = train_df[sampled_indices]
    train_df = train_df.filter(~pl.arange(0, pl.count()).is_in(sampled_indices))

    return train_df, val_df, test_df

def map_article_ids(df: pl.DataFrame, data_path: Path) -> pl.DataFrame:
    """Maps article IDs to new running IDs using a mapping dictionary from JSON.

    Args:
        df (pl.DataFrame): DataFrame with 'article_id' column to be mapped.
        data_path (Path): Path to directory with 'running_id_dict.json' containing ID mappings.

    Returns:
        pl.DataFrame: DataFrame with mapped article IDs, sorted by new IDs. Non-mapped articles are removed.
    """
    with open(data_path / "running_id_dict.json", "r") as f:
        data = json.load(f)
    article_id_dict = data["combined"]

    mapping_df = pl.DataFrame(
        {
            "old_id": list(article_id_dict.keys()),
            "new_id": list(article_id_dict.values()),
        },
        schema_overrides={"old_id": pl.Int32, "new_id": pl.Int32},
    )

    # Join and select
    df = df.join(
        mapping_df, left_on="article_id", right_on="old_id", how="inner"
    ).select(
        pl.col("new_id").alias("article_id"),
        pl.all().exclude(["article_id", "old_id", "new_id"]),
    )
    df = df.sort("article_id")

    return df