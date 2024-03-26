import requests
import urllib
import os
from bs4 import BeautifulSoup

def setup_directory(subdirectory):
    """Set up a download directory within the current working directory."""
    base_dir = os.getcwd()
    download_dir = os.path.join(base_dir, f'Data/{subdirectory}/')
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    return download_dir

def list_file_urls(file_list_url, base_url, extension='.nc', year_filter=None):
    """Retrieve list of file URLs to download, optionally filtering by year."""
    page = requests.get(file_list_url).text
    soup = BeautifulSoup(page, 'html.parser')
    urls = []
    for node in soup.find_all('a'):
        href = node.get('href')
        if href and href.endswith(extension):
            filename = href[href.rfind("/") + 1:]
            # If year filtering is enabled, check if the filename includes a valid year
            if year_filter and not any(str(year) in filename for year in range(year_filter[0], year_filter[1] + 1)):
                continue
            full_url = f"{base_url}{filename}"
            urls.append(full_url)
    return urls


def download_file(url, download_dir):
    """Download a file from a URL into the specified directory."""
    file_name = os.path.join(download_dir, url.split('/')[-1])
    if not os.path.exists(file_name):
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, file_name)
    else:
        print(f"File already exists: {file_name}")


def download_dataset(dataset_info):
    """Download datasets based on the provided information."""
    for info in dataset_info:
        print(f"Starting download for: {info['name']}")
        download_dir = setup_directory(info['subdirectory'])
        file_urls = list_file_urls(info['file_list_url'], info['base_url'], extension=info.get('extension', '.nc'), year_filter=info.get('year_filter'))
        for url in file_urls:
            download_file(url, download_dir)

def main(years):
    datasets = [
        {
            'name': 'SWE',
            'subdirectory': 'SWE',
            'file_list_url': 'https://thredds.met.no/thredds/catalog/senorge/seNorge_snow/swe/catalog.html',
            'base_url': 'https://thredds.met.no/thredds/fileServer/senorge/seNorge_snow/swe/',
            'year_filter': years,
        },
        {
            'name': 'Surface Pressure',
            'subdirectory': 'Pressure',
            'file_list_url': 'https://zenodo.org/record/1970170#.Y9MLlqfMJH4',
            'base_url': 'https://zenodo.org/record/1970170/files/',
            'extension': '?download=1',
            'year_filter': years,
        },
        {
            'name': 'Climate Data',
            'subdirectory': 'Climate',
            'file_list_url': 'https://thredds.met.no/thredds/catalog/senorge/seNorge_2018/Archive/catalog.html',
            'base_url': 'https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Archive/',
            'year_filter': years,
        },
        {
            'name': 'HySN Humidity and Radiation',
            'subdirectory': 'HySN',
            'file_list_url': 'https://zenodo.org/record/1970170#.Y9MLlqfMJH4',
            'base_url': 'https://zenodo.org/record/1970170/files/',
            'extension': '?download=1',
            'year_filter': years,
        }
    ]
    download_dataset(datasets)

if __name__ == "__main__":
    years = (2013, 2015)
    main(years)
