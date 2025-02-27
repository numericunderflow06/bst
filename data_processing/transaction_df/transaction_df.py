from datetime import datetime
from pathlib import Path
import polars as pl

from data_processing.utils.utils_transaction_df import (
    filter_purchases_purchases_per_month_pl,
    map_article_ids,
    train_test_split,
)


def generate_clv_data_pl(
    df: pl.DataFrame,
    agg_df: pl.DataFrame,
    label_threshold: datetime.date,
    pred_end: datetime.date,
    clv_periods: list,
    log_clv: bool = False,
):
    """Generates Customer Lifetime Value (CLV) data from transaction dataframe.

    Args:
        df (pl.DataFrame): Input transaction dataframe containing customer purchases.
        agg_df (pl.DataFrame): Aggregated dataframe containing customer data.
        label_threshold (datetime.date): Start date for CLV calculation period.
        pred_end (datetime.date): End date for CLV calculation period.
        clv_periods (list): List of periods for CLV calculation (currently supports single period only).
        log_clv (bool, optional): Whether to apply log1p transformation to CLV values. Defaults to False.

    Returns:
        pl.DataFrame: Aggregated dataframe with added CLV calculations.

    Raises:
        ValueError: If more than one CLV period is provided.
    """
    if len(clv_periods) > 1:
        raise ValueError("CLV periods should be a single number for now.")

    # Filter transactions between label_threshold and end_date for each period
    filtered_df = df.filter(
        (pl.col("date") >= label_threshold) & (pl.col("date") <= pred_end)
    )

    # Sum total_price for the filtered transactions by customer_id. This is the CLV
    summed_period_df = filtered_df.group_by("customer_id").agg(
        pl.sum("total_price").round(2).alias(f"CLV_label")
    )
    if log_clv:
        summed_period_df = summed_period_df.with_columns(
            pl.col(f"CLV_label").log1p().round(2).alias(f"CLV_label")
        )

    agg_df = agg_df.join(summed_period_df, on="customer_id", how="left")

    agg_df = agg_df.fill_null(0)
    return agg_df


def group_and_convert_df_pl(
    df: pl.DataFrame,
    label_start_date: datetime.date,
    pred_end: datetime.date,
    clv_periods: list,
    cols_to_aggregate: list = [
        "date",
        "days_before",
        "num_items",
        "article_ids",
        "sales_channel_ids",
        "total_price",
        "prices",
    ],
    keep_customer_id: bool = True,
    log_clv: bool = False,
) -> pl.DataFrame:
    """Groups and converts transaction data into aggregated customer-level features.

    Args:
        df (pl.DataFrame): Input transaction dataframe.
        label_start_date (datetime.date): Start date for clv label period.
        pred_end (datetime.date): End date for prediction period.
        clv_periods (list): List of periods for CLV calculation.
        cols_to_aggregate (list, optional): Columns to include in aggregation. Defaults to standard transaction columns.
        keep_customer_id (bool, optional): Whether to retain customer_id in output. Defaults to True.
        log_clv (bool, optional): Whether to apply log1p transformation to CLV values. Defaults to False.

    Returns:
        pl.DataFrame: Aggregated customer-level dataframe.

    Raises:
        ValueError: If required columns (days_before, article_ids, num_items) are missing from cols_to_aggregate.
    """

    if any(
        col not in cols_to_aggregate
        for col in ["days_before", "article_ids", "num_items"]
    ):
        raise ValueError(
            "The columns days_before, article_ids, and num_items are required "
            "for the aggregation"
        )

    mapping = {
        "date": "date_lst",
        "days_before": "days_before_lst",
        "article_ids": "articles_ids_lst",
        "sales_channel_ids": "sales_channel_id_lst",
        "total_price": "total_price_lst",
        "prices": "price_lst",
        "num_items": "num_items_lst",
    }

    agg_df = (
        df.filter(pl.col("date") < label_start_date)
        .with_columns(
            (label_start_date - pl.col("date"))
            .dt.total_days()
            .cast(pl.Int32)
            .alias("days_before"),
            (
                pl.col("sales_channel_ids")
                .cast(pl.List(pl.Int32))
                .alias("sales_channel_ids")
            ),
            pl.col("article_ids").cast(pl.List(pl.Int32)).alias("article_ids"),
        )
        .sort("customer_id", "date")
        .group_by("customer_id")
        .agg(
            pl.col("date").explode().alias("date_lst"),
            pl.col("days_before").explode().alias("days_before_lst"),
            pl.col("article_ids").explode().alias("articles_ids_lst"),
            pl.concat_list(pl.col("sales_channel_ids")).alias("sales_channel_id_lst"),
            pl.col("total_price").explode().alias("total_price_lst"),
            pl.col("prices").explode().alias("price_lst"),
            pl.col("num_items").explode().alias("num_items_lst"),
        )
    )

    if clv_periods is not None:
        agg_df = generate_clv_data_pl(
            df=df,
            agg_df=agg_df,
            label_threshold=label_start_date,
            pred_end=pred_end,
            clv_periods=clv_periods,
            log_clv=log_clv,
        )

    # Drop columns which are not to be aggregated
    cols_to_drop = [v for k, v in mapping.items() if k not in cols_to_aggregate]
    if not keep_customer_id:
        cols_to_drop.append("customer_id")
    agg_df = agg_df.drop(*cols_to_drop)

    return agg_df


def split_df_and_group_pl(
    df: pl.DataFrame,
    clv_periods: list,
    config: dict,
    cols_to_aggregate: list = [
        "date",
        "days_before",
        "article_ids",
        "sales_channel_ids",
        "total_price",
        "prices",
        "num_items",
    ],
    keep_customer_id: bool = True,
    log_clv: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits transaction data into training and test sets and performs aggregation.

    Args:
        df (pl.DataFrame): Input transaction dataframe.
        clv_periods (list): List of periods for CLV calculation.
        config (dict): Configuration dictionary containing:
        cols_to_aggregate (list, optional): Columns to include in aggregation. Defaults to standard transaction columns.
        keep_customer_id (bool, optional): Whether to retain customer_id in output. Defaults to True.
        log_clv (bool, optional): Whether to apply log1p transformation to CLV values. Defaults to False.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing:
            - train_df: Aggregated training dataset
            - test_df: Aggregated test dataset
    """

    train_begin = datetime.strptime(config.get("train_begin"), "%Y-%m-%d")
    train_label_start = datetime.strptime(config.get("train_label_begin"), "%Y-%m-%d")
    train_end = datetime.strptime(config.get("train_end"), "%Y-%m-%d")
    test_begin = datetime.strptime(config.get("test_begin"), "%Y-%m-%d")
    test_label_start = datetime.strptime(config.get("test_label_begin"), "%Y-%m-%d")
    test_end = datetime.strptime(config.get("test_end"), "%Y-%m-%d")

    # Creating the training DataFrame by filtering dates up to `train_end`
    train_df = df.filter(
        (pl.col("date") <= train_end) & (pl.col("date") >= train_begin)
    )

    train_df = group_and_convert_df_pl(
        df=train_df,
        label_start_date=train_label_start,
        pred_end=train_end,
        clv_periods=clv_periods,
        cols_to_aggregate=cols_to_aggregate,
        keep_customer_id=keep_customer_id,
        log_clv=log_clv,
    )

    # Creating the test DataFrame by filtering dates after `test_begin`
    test_df = df.filter((pl.col("date") >= test_begin) & (pl.col("date") <= test_end))

    test_df = group_and_convert_df_pl(
        df=test_df,
        label_start_date=test_label_start,
        pred_end=test_end,
        clv_periods=clv_periods,
        cols_to_aggregate=cols_to_aggregate,
        keep_customer_id=keep_customer_id,
        log_clv=log_clv,
    )

    return train_df, test_df


def load_data_rem_outlier_pl(
    data_path: Path, train_end: datetime.date, group_by_channel_id: bool = False
):
    """Loads transaction data, applies price scaling, and removes outliers.

    Args:
        data_path (Path): Path to directory containing transaction data parquet file.
        train_end (datetime.date): End date for training period.
        group_by_channel_id (bool, optional): Whether to group data by sales channel ID. Defaults to False.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing:
            - grouped_df: Processed transaction dataframe
            - extreme_customers: Dataframe of customers identified as outliers
    """
    file_path = data_path / "transactions_polars.parquet"
    df_pl = pl.read_parquet(file_path)

    df_pl = df_pl.with_columns(
        pl.col("t_dat").alias("date").cast(pl.Date), pl.col("article_id").cast(pl.Int32)
    )

    df_pl = df_pl.with_columns(
        pl.col("price").mul(590).cast(pl.Float32).round(2).alias("price")
    )

    # Map article ids to running ids so that they match with feature matrix
    df_pl = map_article_ids(df=df_pl, data_path=data_path)

    grouped_df, extreme_customers = filter_purchases_purchases_per_month_pl(
        df_pl, train_end=train_end, group_by_channel_id=group_by_channel_id
    )

    return grouped_df, extreme_customers


def get_customer_train_test_articles_pl(
    data_path: Path,
    config: dict,
    clv_periods: list = None,
    cols_to_aggregate: list = [
        "date",
        "days_before",
        "article_ids",
        "sales_channel_ids",
        "total_price",
        "prices",
        "num_items",
    ],
    keep_customer_id: bool = True,
):
    """Processes customer transaction data into train and test sets with article information.

    Args:
        data_path (Path): Path to directory containing transaction data.
        config (dict): Configuration dictionary for data processing parameters.
        clv_periods (list, optional): List of periods for CLV calculation. Defaults to None.
        cols_to_aggregate (list, optional): Columns to include in aggregation. Defaults to standard transaction columns.
        keep_customer_id (bool, optional): Whether to retain customer_id in output. Defaults to True.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing:
            - train_df: Processed training dataset with article information
            - test_df: Processed test dataset with article information
    """
    train_end = datetime.strptime(config.get("train_end"), "%Y-%m-%d")
    grouped_df, extreme_customers = load_data_rem_outlier_pl(
        data_path=data_path, train_end=train_end
    )

    train_df, test_df = split_df_and_group_pl(
        df=grouped_df,
        clv_periods=clv_periods,
        config=config,
        cols_to_aggregate=cols_to_aggregate,
        keep_customer_id=True,
        log_clv=config.get("log_clv", False),
    )

    train_df = train_df.join(extreme_customers, on="customer_id", how="anti")
    test_df = test_df.join(extreme_customers, on="customer_id", how="anti")

    if not keep_customer_id:
        train_df = train_df.drop("customer_id")
        test_df = test_df.drop("customer_id")

    return train_df, test_df


def get_tx_article_dfs(
    data_path: Path,
    config: dict,
    cols_to_aggregate: list = [
        "date",
        "days_before",
        "article_ids",
        "sales_channel_ids",
        "total_price",
        "prices",
        "num_items",
    ],
    keep_customer_id: bool = True,
):
    """Creates train, validation, and test datasets with optional subsampling.

    Args:
        data_path (Path): Path to directory containing transaction data files.
        config (dict): Configuration dictionary containing:
        cols_to_aggregate (list, optional): Transaction columns to include in output.
        keep_customer_id (bool, optional): Whether to retain customer_id column.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Tuple containing:
            - train_df: Final training dataset (subset of original training data)
            - val_df: Validation dataset (10% of original training data)
            - test_df: Test dataset (optionally subsampled)
    """
    """
    Columns of dfs:
        - customer_id
        - date_lst (list[date]): Dates of each transaction
        - days_before_lst (list[int]): Number of days between start of prediction and date of transction
        - articles_ids_lst (list[int]): Flattened list of all items a customer purchased 
        - sales_channel_id_lst (list[list[int]]): Sales channel of a transaction (repeated for each item within a transaction)
        - total_price_lst (list[float]): Value of each transaction
        - price_lst (list[float]): Flattened list of prices of all items customer purchased
        - num_items_lst (list[int]): Number of items in each transaction
        - CLV_label (float): Sales in prediction period (label to be used)
    """
    train_df, test_df = get_customer_train_test_articles_pl(
        data_path=data_path,
        config=config,
        clv_periods=config.get("clv_periods", [6]),
        cols_to_aggregate=cols_to_aggregate,
        keep_customer_id=keep_customer_id,
    )
    train_df, val_df, test_df = train_test_split(
        train_df=train_df,
        test_df=test_df,
        subset=config.get("subset"),
        train_subsample_percentage=config.get("train_subsample_percentage"),
    )
    return train_df, val_df, test_df
