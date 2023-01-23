import pickle
import boto3
import yaml
import io
import pandas as pd
from botocore.errorfactory import ClientError
from typing import Dict, Any


# def save_object(object: Any, path: str, credentials: Dict[str, str]):
#     """
#     Save an object to AWS s3 as pickle file.

#     Args:
#         object (Any): Any object that could be saved as pickle file
#         path (str): Path to save the object. You dont't need to specify
#         bucket name here
#         credentials (Dict[str, str]): AWS credentials containing the following
#         fields under 'aws' section: 'region', 'access_key', 'secret_access_key'
#     """
#     s3 = boto3.resource(
#         's3',
#         region_name=credentials['aws']['region'],
#         aws_access_key_id=credentials['aws']['access_key'],
#         aws_secret_access_key=credentials['aws']['secret_access_key']
#     )
#     byte_obj = pickle.dumps(object)
#     s3.Object(credentials['aws']['bucket'], path).put(Body=byte_obj)


# def load_object(path: str, credentials: Dict[str, str]) -> Any:
#     """
#     Load an object from AWS s3.

#     Args:
#         path (str): Path to save the object. You dont't need to specify
#         bucket name here
#         credentials (Dict[str, str]): AWS credentials containing the following
#         fields under 'aws' section: 'region', 'access_key', 'secret_access_key'
#     """
#     s3 = boto3.resource(
#         's3',
#         region_name=credentials['aws']['region'],
#         aws_access_key_id=credentials['aws']['access_key'],
#         aws_secret_access_key=credentials['aws']['secret_access_key']
#     )
#     response = s3.Object(credentials['aws']['bucket'], path).get()
#     obj_bytes = response['Body'].read()
#     return pickle.loads(obj_bytes)


def check_file(s3_client, bucket: str, key: str) -> bool:
    try:
        _ = s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def load_yaml(path: str, *sections) -> Any:
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    for s in sections:
        d = d[s]
    return d


def write_df_to_s3(
    s3_client,
    data: pd.DataFrame,
    bucket: str,
    key: str,
    format: str,
    **write_kwargs
):
    out_buffer = io.BytesIO()
    if format == 'parquet':
        data.to_parquet(out_buffer, index=False, **write_kwargs)
    elif format == 'csv':
        data.to_csv(out_buffer, index=False, **write_kwargs)
    else:
        raise ValueError('Unsupported format argument')

    s3_client.put_object(
        Body=out_buffer.getvalue(),
        ContentType='application/vnd.ms-excel',
        Bucket=bucket,
        Key=key
    )


def read_df_from_s3(
    s3_client,
    bucket: str,
    key: str,
    format: str,
    **read_kwargs
) -> pd.DataFrame:
    """
    Read data from S3 and load it into Pandas DataFrame.

    Args:
        s3_client: S3 Client
        bucket (str): S3 Bucket
        key (str): Path in S3 bucket to save file
        format (str): The function supports only 'csv' and 'parquet'
    Returns:
        pd.DataFrame: data upates records
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    if format == 'csv':
        return pd.read_csv(obj['Body'], **read_kwargs)
    elif format == 'parquet':
        return pd.read_parquet(obj['Body'], **read_kwargs)
    else:
        raise ValueError('Unsupported format argument')


def check_file(s3_client, bucket: str, key: str) -> bool:
    try:
        _ = s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
