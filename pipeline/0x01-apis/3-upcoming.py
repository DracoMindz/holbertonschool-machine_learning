#!/usr/bin/env python3
""" Write script that displays the up coming launch"""

import requests

if __name__ == '__main__':
    # need url
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    # request url
    launchData = requests.get(url).json()
    # set date type for unbound value comparison
    date = float('inf')
    # set for loop to iterate through launches
    for idx, launch in enumerate(launchData):
        if date > launch["date_unix"]:
            date = launch["date_unix"]
            launchIdx = idx
    # get data based on launchData[launchIdx]
    upLaunch = launchData[launchIdx]
    # name
    name = upLaunch["name"]
    # date in local time
    dateLocal = upLaunch["date_local"]
    # rocket: need url, request.get, json, rocket name
    rocket = upLaunch["rocket"]
    rocketUrl = "https://api.spacexdata.com/v4/rockets/{}".format(rocket)
    rocketData = requests.get(rocketUrl).json()
    rocketName = rocketData["name"]

    # launch pad: need url, request.get, json, launch pad name
    launchPad = upLaunch["launchpad"]
    launchPadUrl = "https://api.spacexdata.com/v4/launchpads/{}".\
        format(launchPad)
    launchPadData = requests.get(launchPadUrl).json()
    launchPadName = launchPadData["name"]

    # launch pad locality
    launchPadLoc = launchPadData["locality"]

    # Print name, (date in local time), rocket name,
    # launch pad name, (lunch pad locality)
    print("{} ({}) {} - {} ({})".format(name, dateLocal,
                                        rocketName, launchPadName,
                                        launchPadLoc))
