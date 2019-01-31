import tempfile

from google.cloud import storage


def download_file_from_gcs(project, bucket_name, file_location, destination_file_location=None):
    """
    Downloads a file from Google Cloud Storage
    Returns the location where the file was saved locally
    """
    gs_location = 'gs://{}/{}'.format(bucket_name, file_location)

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name)

    blob = bucket.get_blob(file_location)
    if not blob:
        raise FileNotFoundError('{} does not exist!'.format(gs_location))

    if not destination_file_location:
        destination_file_location = tempfile.NamedTemporaryFile(delete=False).name

    blob.chunk_size = 1 << 29 # Increased chunk size for faster downloading
    blob.download_to_filename(destination_file_location)

    return destination_file_location
