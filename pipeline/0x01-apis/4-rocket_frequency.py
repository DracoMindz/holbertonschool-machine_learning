#!/usr/bin/env python3
""" Write a script that displays the num of launches per rocket"""

import requests


if __name__ == '__main__':
    # need url
    url = "https://api.spacexdata.com/v4/launches/"
    rocketLaunches = {}
    # request url
    launchData = requests.get(url).json()
    # set for loop to iterate through launches
    for idx, launch in enumerate(launchData):
        # rocket: need url, request.get, json, rocket name
        rocket = launch["rocket"]
        rocketUrl = "https://api.spacexdata.com/v4/rockets/{}".format(rocket)
        rocketData = requests.get(rocketUrl).json()
        rocketName = rocketData["name"]
        if rocketName in rocketLaunches:
            # advance to next
            rocketLaunches[rocketName] = +1
        if rocketName not in rocketLaunches:
            rocketLaunches[rocketName] = 1
        # sorting conditions: sort dicts
    rocketsUp = sorted(rocketLaunches.keys(), key=lambda kv: kv[0])
    rocketsUp = sorted(rocketsUp, key=lambda kv: kv[1], reverse=True)
    # Print rocket name, num rockets
    for rocketName in rocket:
        print("{}:{}".format(rocketsUp[0], rocketsUp[1]))
