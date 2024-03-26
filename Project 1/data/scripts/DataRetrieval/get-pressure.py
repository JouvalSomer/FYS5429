#Retrieve HySN humidity and radiation datasets
"""
Download surface pressure. Restric to 1980 to 2017 needs to be added. This was the Humidity file copied and changed
Some variable names might not make sense.
"""
import requests
import urllib
import os
from bs4 import BeautifulSoup

#Location where data files are stored
Directory = os.getcwd()
print("script started")
Download_Dir = os.path.join(Directory, 'Data/Pressure/')
# Create the directory if it doesn't exist
if not os.path.exists(Download_Dir):
    os.makedirs(Download_Dir)
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

#Filter for only humidity and shortwave radition files.
filelist_Humidity = [k for k in filelist if 'Pressure' in k]


print(len(filelist_Humidity))

#Loop to designate filenames and directory then download humidity files
for file in filelist_Humidity[:1]:
    Download_Dir = os.path.join(Directory, 'Data/Pressure/')
    download_url(file)
