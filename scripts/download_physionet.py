import os
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from html.parser import HTMLParser

# Command-line argument setup
parser = argparse.ArgumentParser(description='Download files recursively from a given URL.')
parser.add_argument('url', help='The URL to download from.')
parser.add_argument('username', help='The username for authentication.')
parser.add_argument('password', help='The password for authentication.')
parser.add_argument('--download-dir', default='.', help='The directory to save the downloaded files.')
args = parser.parse_args()

# HTML parser to find links
class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.files = []
        self.directories = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href', '')
            if href.endswith('/'):
                self.directories.append(href)
            else:
                self.files.append(href)

def download_file(url):
    local_filename = url.split('/')[-1]
    local_path = os.path.join(args.download_dir, local_filename)
    with requests.get(url, auth=(args.username, args.password), stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def list_files(url):
    response = requests.get(url, auth=(args.username, args.password))
    parser = LinkParser()
    parser.feed(response.text)
    return [urljoin(url, file) for file in parser.files], [urljoin(url, directory) for directory in parser.directories]

def main():
    os.makedirs(args.download_dir, exist_ok=True)
    files_to_download, directories_to_traverse = list_files(args.url)

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(download_file, files_to_download)

        # Recursive call for directories
        for directory in directories_to_traverse:
            if directory != '../':  # Ignore the parent directory link
                main.url = directory  # Update the URL for recursive call
                main()

if __name__ == "__main__":
    main.url = args.url  # Initial URL for the main function
    main()
