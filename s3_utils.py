# s3_utils.py
import boto3
import os
import time
import mimetypes
import io
from botocore.exceptions import BotoCoreError, ClientError


def upload_to_s3(file_data: bytes, original_filename: str, bucket_name: str):
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("S3_BUCKET_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_BUCKET_SECRET_KEY"),
        )

        # Determine file extension
        file_extension = mimetypes.guess_extension(mimetypes.guess_type(original_filename)[0]) or '.webp'
        new_file_name = f"file_{int(time.time())}{file_extension}"

        # Upload file to S3
        s3.upload_fileobj(
            Fileobj=io.BytesIO(file_data),
            Bucket=bucket_name,
            Key=new_file_name
        )

        # Return the S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{new_file_name}"
        return {"Location": s3_url, "Key": new_file_name}

    except (BotoCoreError, ClientError) as e:
        print(f"S3 Upload Error: {e}")
        raise e


def generate_signed_url(file_name: str, bucket_name: str, expiration: int = 60):
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("S3_BUCKET_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_BUCKET_SECRET_KEY"),
        )

        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': file_name
            },
            ExpiresIn=expiration
        )
        return url

    except Exception as e:
        print(f"S3 Signed URL Error: {e}")
        return None
