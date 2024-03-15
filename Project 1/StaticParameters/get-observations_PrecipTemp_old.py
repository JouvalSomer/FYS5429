#Script to pull seNorge 2018 daily precipitation and temperature netcdf data

import requests
import urllib
import os
from bs4 import BeautifulSoup

print("Srcipt started")

#Location where the files are downloaded
Directory = os.getcwd()
Download_Dir = os.path.join(Directory, 'Data/Climate/')
print("Directory: ",Directory)

#Partial url address of where to download files but need to add file names to the end of it
url = "https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Archive/"
#This is the url address of where the files names are for all the data we need to download
FList = "https://thredds.met.no/thredds/catalog/senorge/seNorge_2018/Archive/catalog.html"
#This tells the function which type of files to look for


#Defining function to retrieve list of filenames to download and to generate download url addresses. Requires the designated inputs above
def listFD(FList, url):
    page = requests.get(FList).text
    soup = BeautifulSoup(page, 'html.parser')
    temp = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.nc')]
    temp2 = [elem[elem.rfind("/") + 1:] for elem in temp]
    return [url + s for s in temp2]

#Defining function to generate filename from url and download directory and to download files using urllib
def download_url(url):
    print("downloading: ",url)
    # assumes that the last segment after the / represents the file name
    file_name_start_pos = url.rfind("/") + 1
    file_name = os.path.join(Download_Dir, url[file_name_start_pos:])
    print(file_name)
    urllib.request.urlretrieve(str(url), file_name)

#Generating empty list to for loop that retieves filenames to download
filelist = []

#Loop to retrieve filenames and generate url address to download
for file in listFD(FList, url):
    filelist.append(file)

#Subsetting filelist for testing
filelist = filelist[34:45:5]
print(filelist)
#Loop to designate filenames & directory and download files
for file in filelist:
    download_url(file)