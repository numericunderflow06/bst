from pathlib import Path
from data_processing.customer_df.customer_df import get_customer_df_benchmarks
from data_processing.transaction_df.transaction_df import get_tx_article_dfs
import polars as pl


def expand_list_columns(
    df: pl.DataFrame, date_col: str = "days_before_lst", num_col: str = "num_items_lst"
) -> pl.DataFrame:
    """
    Expand a Polars DataFrame by repeating each element in a list column according to
    the counts specified in another list column.

    Args:
        df: Input Polars DataFrame with list columns
        date_col: Name of the column containing the lists to be expanded
        num_col: Name of the column containing lists of counts

    Returns:
        A new Polars DataFrame where the list elements in date_col have been expanded
    """
    expanded = df.with_columns(
        pl.struct([date_col, num_col])
        .map_elements(
            lambda x: [
                date
                for date, count in zip(x[date_col], x[num_col])
                for _ in range(count)
            ]
        )
        .alias(date_col)
    )

    return expanded


def add_benchmark_tx_features(df: pl.DataFrame) -> pl.DataFrame:
    """Creates benchmark transaction features from aggregated customer transaction data.

    Args:
        df: A Polars DataFrame containing aggregated transaction data with list columns
            including total_price_lst, num_items_lst, days_before_lst, price_lst,
            and CLV_label.

    Returns:
        pl.DataFrame: A DataFrame with derived features including:
            - total_spent: Sum of all transaction amounts
            - total_purchases: Count of transactions
            - total_items: Sum of items purchased
            - days_since_last_purchase: Days since most recent transaction
            - days_since_first_purchase: Days since first transaction
            - avg_spent_per_transaction: Mean transaction amount
            - avg_items_per_transaction: Mean items per transaction
            - avg_days_between: Mean days between transactions
            - regression_label: CLV label for regression
            - classification_label: Binary CLV label (>0)

    Note:
        The avg_days_between calculation may return None for customers with single
        transactions, which is handled by tree-based algorithms.
    """
    return df.select(
        "customer_id",
        pl.col("total_price_lst").list.sum().alias("total_spent"),
        pl.col("total_price_lst").list.len().alias("total_purchases"),
        pl.col("num_items_lst").list.sum().alias("total_items"),
        pl.col("days_before_lst").list.get(-1).alias("days_since_last_purchase"),
        pl.col("days_before_lst").list.get(0).alias("days_since_first_purchase"),
        pl.col("price_lst").list.mean().alias("avg_spent_per_transaction"),
        (
            pl.col("num_items_lst")
            .list.mean()
            .cast(pl.Float32)
            .alias("avg_items_per_transaction")
        ),
        # Code below returns None values for customers with single Tx
        # Tree algos should be able to handle this
        (
            pl.col("days_before_lst")
            .list.diff(null_behavior="drop")
            .list.mean()
            .mul(-1)
            .cast(pl.Float32)
            .alias("avg_days_between")
        ),
        pl.col("CLV_label").alias("regression_label"),
        pl.col("CLV_label").gt(0).cast(pl.Int32).alias("classification_label"),
    )


def process_dataframe(df: pl.DataFrame, max_length: int = 20) -> pl.DataFrame:
    """Processes a polars DataFrame by expanding list columns and selecting specific columns with transformations.

    This function performs several operations on the input DataFrame:
    1. Expands list columns using the expand_list_columns function
    2. Selects and renames specific columns
    3. Truncates list columns to a maximum length

    Args:
        df: A polars DataFrame containing customer transaction data
        max_length: Maximum number of elements to keep in list columns (default: 20)

    Returns:
        A processed polars DataFrame with the following columns:
            - customer_id: Customer identifier
            - days_before_lst: Truncated list of days before some reference date
            - articles_ids_lst: Truncated list of article identifiers
            - regression_label: CLV label for regression tasks
            - classification_label: Binary classification label derived from CLV
    """
    df = expand_list_columns(df, date_col="days_before_lst", num_col="num_items_lst")
    return df.select(
        "customer_id",
        "days_before_lst",
        "articles_ids_lst",
        pl.col("CLV_label").alias("regression_label"),
        pl.col("CLV_label").gt(0).cast(pl.Int32).alias("classification_label"),
    ).with_columns(
        pl.col("days_before_lst").list.tail(max_length),
        pl.col("articles_ids_lst").list.tail(max_length),
    )


def get_benchmark_dfs(
    data_path: Path, config: dict
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Creates benchmark train, validation, and test datasets with transaction and customer features.

    Args:
        data_path: Path object pointing to the data directory
        config: Dictionary containing configuration parameters for data processing

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
            - train_df: Training dataset with benchmark features
            - val_df: Validation dataset with benchmark features
            - test_df: Test dataset with benchmark features

        Each DataFrame contains transaction-derived features joined with customer features.
    """
    train_article, val_article, test_article = get_tx_article_dfs(
        data_path=data_path,
        config=config,
        cols_to_aggregate=[
            "date",
            "days_before",
            "article_ids",
            "sales_channel_ids",
            "total_price",
            "prices",
            "num_items",
        ],
        keep_customer_id=True,
    )

    customer_df = get_customer_df_benchmarks(data_path=data_path, config=config)

    train_df = process_dataframe(
        df=train_article, max_length=config["max_length"]
    ).join(customer_df, on="customer_id", how="left")
    val_df = process_dataframe(df=val_article, max_length=config["max_length"]).join(
        customer_df, on="customer_id", how="left"
    )
    test_df = process_dataframe(df=test_article, max_length=config["max_length"]).join(
        customer_df, on="customer_id", how="left"
    )

    return train_df, val_df, test_df
