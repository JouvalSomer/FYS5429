"""
Download Climate data. Relevant parameters are precipitation, min and max temperature. It would also make sense
to restrict this to only download the specifc years from 1980 to 2017

It is named old, because the PhD named it like this.

TODO:
- download only dates from 1980 to 2017
"""


import requests
import urllib
import os
from bs4 import BeautifulSoup

print("Script started")

# Location where the files are downloaded
Directory = os.getcwd()
Download_Dir = os.path.join(Directory, 'Data/Climate/')
print("Directory:", Directory)
# Create the directory if it doesn't exist
if not os.path.exists(Download_Dir):
    os.makedirs(Download_Dir)

# Partial URL address of where to download files but need to add file names to the end of it
url = "https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Archive/"
# This is the URL address of where the files names are for all the data we need to download
FList = "https://thredds.met.no/thredds/catalog/senorge/seNorge_2018/Archive/catalog.html"
# This tells the function which type of files to look for


# Defining function to retrieve list of filenames to download and to generate download URL addresses. Requires the designated inputs above
def listFD(FList, url):
    page = requests.get(FList).text
    soup = BeautifulSoup(page, 'html.parser')
    temp = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.nc')]
    temp2 = [elem[elem.rfind("/") + 1:] for elem in temp]
    return [url + s for s in temp2]

# Defining function to generate filename from URL and download directory and to download files using urllib
def download_url(url):
    # Assumes that the last segment after the / represents the file name
    file_name_start_pos = url.rfind("/") + 1
    file_name = os.path.join(Download_Dir, url[file_name_start_pos:])
    # Check if the file already exists in the directory
    if not os.path.exists(file_name):
        print("Downloading:", url)
        print("Saving as:", file_name)
        urllib.request.urlretrieve(str(url), file_name)
    else:
        print("File", file_name, "already exists. Skipping download.")

# Generating empty list to for loop that retrieves filenames to download
filelist = []

# Loop to retrieve filenames and generate URL address to download
for file in listFD(FList, url):
    # Extract the year from the filename
    year = file[-7:-3]
    print(year)
    # Check if the year falls within the range 1980 to 2017
    if 1980 <= int(year) <= 2017:
        filelist.append(file)

# Loop to designate filenames & directory and download files
for file in filelist[:1]:
    download_url(file)

print("Download completed.")