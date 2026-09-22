import pyarrow.parquet as pq


def get_parquet_metadata(file_path):
    """
    Return basic metadata of a Parquet file.
    """

    parquet_file = pq.ParquetFile(file_path)

    return {
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": parquet_file.metadata.num_columns,
        "num_row_groups": parquet_file.num_row_groups,
        "columns": parquet_file.schema.names,
    }


def read_parquet_columns(file_path, columns=None):
    """
    Read selected columns from a Parquet file.

    Parameters
    ----------
    file_path : str
        Path to the Parquet file.

    columns : list, optional
        Columns to load.

    Returns
    -------
    pandas.DataFrame
        Selected data.
    """

    return pq.read_table(
        file_path,
        columns=columns
    ).to_pandas()


def read_parquet_row_group(file_path, row_group, columns=None):
    """
    Read one row group from a Parquet file.

    This is useful for memory-efficient processing
    of large datasets.
    """

    parquet_file = pq.ParquetFile(file_path)

    return parquet_file.read_row_group(
        row_group,
        columns=columns
    ).to_pandas()
