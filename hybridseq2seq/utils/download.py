import os
import requests
from tqdm import tqdm


def download(url: str, path: str, ignore_if_exists: bool = True):
    """
    Downloads a URL to a given path on disk
    """
    if os.path.exists(path) and ignore_if_exists:
        return

    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print(
            "Exception when trying to download {}. Response {}".format(
                url, req.status_code
            ),
            file=sys.stderr,
        )
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()
