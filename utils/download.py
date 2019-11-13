__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"
import requests
import tqdm  # progress bar
import os.path


def from_url(url, path=None, file=None):
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, file)
    else:
        filename = os.path.join(".", url.split('/')[-1])

    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    # chunk = 1
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)

    with open(filename, 'wb') as fp:
        for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=chunk_size)
                , total=num_bars
                , unit='KB'
                , desc=filename
                , leave=True  # progressbar stays
        ):
            fp.write(chunk)

        fp.close()

    return filename
