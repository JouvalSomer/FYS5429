#Retrieve HySN humidity and radiation datasets
"""
One thing to add would be to restric files that are getting downloaded to the years 1980-2017 including both.
"""
import requests
import urllib
import os
from bs4 import BeautifulSoup

#Location where data files are stored
Directory = os.getcwd()
print("script started")

#Partial url address of where to download files but need to add file names to the end of it
url = "https://zenodo.org/record/1970170/files/"
#This is the url address of where the files names are for all the data we need to download
FList = "https://zenodo.org/record/1970170#.Y9MLlqfMJH4"


#Defining function to retrieve list of filenames to download and to generate download url addresses. Requires the designated inputs above
def listFD(FList, url):
    page = requests.get(FList).text
    soup = BeautifulSoup(page, 'html.parser')
    temp = [node.get('href') for node in soup.find_all('link') if node.get('href').endswith('.nc')]
    temp2 = [elem[elem.rfind("/") + 1:] for elem in temp]
    return [url + s + '?download=1' for s in temp2]

#Defining function to generate filename from url and download directory and to download files using urllib
def download_url(url):
  print("downloading: ",url)
  # assumes that the last segment after the / represents the file name
  # if url is abc/xyz/file.txt, the file name will be file.txt
  file_name_start_pos = url.rfind("/") + 1
  file_name = os.path.join(Download_Dir, url[url.rfind("/") + 1:len(url)-11])
  print(file_name)
  urllib.request.urlretrieve(str(url), file_name)

#Generating empty list to for loop that retieves filenames to download
filelist = []

#Loop to retrieve filenames and generate url address to download
for file in listFD(FList, url):
    filelist.append(file)

#Filter for only humidity and shortwave radition files.
filelist_Humidity = [k for k in filelist if 'Humidity' in k]
filelist_SW_Rad = [k for k in filelist if 'Shortwave' in k]

print(len(filelist_Humidity))

#Loop to designate filenames and directory then download humidity files
for file in filelist_Humidity:
    Download_Dir = os.path.join(Directory, 'Data/Humidity/')
    download_url(file)

#Loop to designate filenames and directory then download shortwave radiation files
for file in filelist_SW_Rad:
    Download_Dir = os.path.join(Directory, 'Data/Radiation/')
    download_url(file)