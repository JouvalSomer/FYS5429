#Script to pull seNorge 2018 daily precipitation and temperature parameters netcdf data
"""
Download Snow water equvialent. Restric to 1980 to 2017 needs to be added.
"""
import requests
import urllib
import os
from bs4 import BeautifulSoup

print("scirpt started")
#Location where the files are downloaded
Directory = os.getcwd()
Download_Dir = os.path.join(Directory, 'Data/SWE/')
# Create the directory if it doesn't exist
if not os.path.exists(Download_Dir):
    os.makedirs(Download_Dir)

print(Download_Dir)
#Partial url address of where to download files but need to add file names to the end of it
url = "https://thredds.met.no/thredds/fileServer/senorge/seNorge_snow/swe/"
#This is the url address of where the files names are for all the data we need to download
FList = "https://thredds.met.no/thredds/catalog/senorge/seNorge_snow/swe/catalog.html"
#This tells the function which type of files to look for
ext = '.nc'

#Defining function to retrieve list of filenames to download and to generate download url addresses. Requires the designated inputs above
def listFD(FList, url, ext=''):
    page = requests.get(FList).text
    soup = BeautifulSoup(page, 'html.parser')
    temp = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    temp2 = [elem[elem.rfind("/") + 1:] for elem in temp]
    return [url + s for s in temp2]

#Defining function to generate filename from url and download directory and to download files using urllib
def download_url(url):
  print("downloading: ",url)
  # assumes that the last segment after the / represents the file name
  # if url is abc/xyz/file.txt, the file name will be file.txt
  file_name_start_pos = url.rfind("/") + 1
  file_name = os.path.join(Download_Dir, url[file_name_start_pos:])
  print(file_name)
  urllib.request.urlretrieve(str(url), file_name)

#Generating empty list to for loop that retieves filenames to download
filelist = []

#Loop to retrieve filenames and generate url address to download
for file in listFD(FList, url, ext):
    # Extract the year from the filename
    if "latest" not in file:
        # Extract the year from the filename
        # Split the URL by "/" to isolate the filename
        #print("before splitt /",file)
        year = file.split("/")[-1]
        # Extract the year portion from the filename
        year = year.split("_")[-1].split(".")[0]
        print(year)
        # Check if the year falls within the range 1980 to 2017
        if 1980 <= int(year) <= 2017:
            filelist.append(file)

#Loop to designate filenames & directory and download files
for file in filelist[:1]:
    download_url(file)