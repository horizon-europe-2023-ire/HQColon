import gzip
import shutil


def decompress_gzip(compressed_path, target_path):
    """Decompress a gzipped file."""
    try:
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File decompressed and saved as {target_path}")
    except Exception as e:
        print(f"Error during decompression: {e}")


def move_file(source_path, destination_path):
    """Move a file from source to destination."""
    try:
        shutil.copy(source_path, destination_path)
        print(f"File moved to {destination_path}")
    except Exception as e:
        print(f"Error during moving the file: {e}")
