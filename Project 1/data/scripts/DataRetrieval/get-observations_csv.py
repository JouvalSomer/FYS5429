#!/usr/bin/python3

"""
Official Script from NVE for getting hydrological discharge data from NVE stations.

Change I did: The output folder is now stationID dependent, because the first ID indicates which catchment




"""

import pandas as pd
import getopt
import json
import sys
import os


from urllib.request import Request, urlopen  # Python 3
#except ImportError:
#    from urllib2 import Request, urlopen  # Python 2

print("Script started")
def usage():
    print()
    print("Get observations from the NVE Hydrological API (HydAPI)")
    print("Parameters:")
    print("   -a: ApiKey (mandatory). ")
    print("   -s: StationId (mandatory). Several stations can be given separated by comma. Example \"6.10.0,12.209.0")
    print("   -p: Parameter (mandatory). Several Parameters can be given se")
    print("   -r: Resolution time. 0 (instantenous),60 (hourly) or 1440 (daily). (mandatory)")
    print("   -t: Reference time. See documentation for referencetime. Example \"P1D/\", gives one day back in time. If none given, the last observed value will be returned")
    print("   -h: This help")
    print()
    print("Example:")
    print("    python get-observations.py -a \"INSERT_APIKEY_HERE\" -s \"6.10.0,12.209.0\" -p \"1000,1001\" -r 60 -t \"P1D/\"")
    print()


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "a:s:p:r:ht:")
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        print("exitin")
        sys.exit(2)

    station = None
    parameter = None
    resolution_time = None
    api_key = None
    reference_time = None

    for opt, arg in opts:
        if opt == "-s":
            station = arg
            print(type(station))
        elif opt == "-p":
            parameter = arg
        elif opt == "-r":
            resolution_time = arg
        elif opt == "-a":
            api_key = arg
        elif opt == "-t":
            reference_time = arg
        elif opt == "-h":
            usage()
            print("exiting")
            sys.exit()
        else:
            assert False, "unhandled option"

    if api_key == None:
        print("Error: You must supply the api-key with your request (-a)")
        usage()
        print("exit api")
        sys.exit(2)

    if station == None or parameter == None or resolution_time == None:
        print("Error: You must supply the parameters station (-s), parameter (-p) and resolution time (-r)")
        usage()
        print("exit station")
        sys.exit(2)
    baseurl = "https://hydapi.nve.no/api/v1/Observations?StationId={station}&Parameter={parameter}&ResolutionTime={resolution_time}"

    url = baseurl.format(station=station, parameter=parameter,
                         resolution_time=resolution_time)

    if reference_time is not None:
        url = "{url}&ReferenceTime={reference_time}".format(
            url=url, reference_time=reference_time)

    print(url)

    request_headers = {
        "Accept": "application/json",
        "X-API-Key": api_key
    }

    request = Request(url, headers=request_headers)

    response = urlopen(request)
    content = response.read().decode('utf-8')

    
    parsed_result = json.loads(content)
    # json_data = json.dumps(parsed_result['data'])
    # parsed_result['data'][0] is a dict
    #print(parsed_result['data'][0].keys())
    df = pd.read_json(json.dumps(parsed_result['data'][0]['observations']))
    station_id = station.replace(".", "_")
    print(station_id)
    print(station_id.split("_",1)[0])
    Directory = os.getcwd()
    Obs_file = os.path.join(Directory, "Data/Streamflow/NVE_stations/Catchment"+ station_id.split("_",1)[0],"NVEObservation"+ station_id + '.cvs')
    #Obs_file = os.path.join(Directory, "Data/Streamflow/NVE_stations/NVEObservation"+ station_id + ".cvs")
    print(Obs_file)

    # Extract the directory part of the path
    directory_path = os.path.dirname(Obs_file)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    df.to_csv(Obs_file, index=False)



if __name__ == "__main__":
    main(sys.argv[1:])

print("getobservations_CSV Done")
