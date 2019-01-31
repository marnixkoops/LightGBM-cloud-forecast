from google.cloud import storage
from google.cloud.storage.bucket import Bucket


def upload_file_to_gcs(project, bucket, file_location, destination_file_location):  # pragma no cover
    """

    Uploads the the file at "file_location" to the "destination_file_location" on the Google Cloud Storage "bucket"
    """
    if type(bucket) is not Bucket:
        client = storage.Client(project=project)
        bucket = client.get_bucket(bucket)

    blob = bucket.blob(destination_file_location)
    blob.chunk_size = 1 << 29 # Increased chunk size for faster uploading
    blob.upload_from_filename(file_location)

    return True
