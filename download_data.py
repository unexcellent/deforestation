import argparse
import logging
from pathlib import Path
from typing import TypedDict
from tqdm import tqdm

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class S3Object(TypedDict):
    Key: str
    Size: int


def get_s3_objects(s3: any, bucket: str, prefix: str) -> list[S3Object]:
    paginator = s3.get_paginator("list_objects_v2")
    objects = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            objects.extend(page["Contents"])

    return objects


def download_s3_folder(
    bucket_name: str, folder_name: str, local_dir: str = "./data", reverse: bool = False
) -> None:
    """Downloads a specific folder from an S3 bucket with configurable ordering."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = f"{folder_name.strip('/')}/" if folder_name.strip("/") else ""

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        objects = get_s3_objects(s3, bucket_name, prefix)

        if not objects:
            logger.warning(
                f"No objects found in '{folder_name}' within bucket '{bucket_name}'"
            )
            return

        # Sort objects by Key. Reverse=True moves 'end' files to the start.
        sorted_objects = sorted(objects, key=lambda x: x["Key"], reverse=reverse)

        pbar = tqdm(sorted_objects)
        for obj in pbar:
            key = obj["Key"]
            target = local_path / key
            pbar.desc = key.split("/")[-1]

            if key.endswith("/") or key == prefix:
                continue

            if target.exists():
                continue

            if "sentinel-2" not in str(target) and "label" not in str(target):
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket_name, key, str(target))

    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        raise
    except ClientError as e:
        logger.error(f"AWS client error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a folder from S3 with ordering control"
    )
    parser.add_argument(
        "--bucket_name",
        default="osapiens-terra-challenge",
        help="Name of the S3 bucket",
    )
    parser.add_argument(
        "--folder_name",
        default="makeathon-challenge",
        help="Folder inside the S3 bucket",
    )
    parser.add_argument(
        "--local_dir", default="./data", help="Local directory to save files"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Download files in reverse alphabetical order",
    )

    args = parser.parse_args()

    download_s3_folder(args.bucket_name, args.folder_name, args.local_dir, args.reverse)
