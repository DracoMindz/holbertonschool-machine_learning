#!/usr/bin/env python3
""" Write SCript  that prints location of user"""

import requests
import sys
import time


if __name__ == '__main__':
    # need arguement
    url = sys.argv[1]
    # set key:value for payload
    payload = {'accept': 'application/vnd.github.v3+json'}
    # request url with params=payload
    r = requests.get(url, params=payload)
    # status code == 403
    if r.status_code == 403:
        # set value for x / 60
        limit = (int(r.headers["X-Ratelimit-Reset"]))
        x = (limit - int(time.time()))
        x = (int(x / 60))if __name__ == '__main__':
    # need arguement
    url = sys.argv[1]
    # set key:value for payload
    payload = {'accept': 'application/vnd.github.v3+json'}
    # request url with params=payload
    r = requests.get(url, params=payload)
    # status code == 403
    if r.status_code == 403:
        # set value for x / 60
        limit = (int(r.headers["X-Ratelimit-Reset"]))
        x = (limit - int(time.time()))
        x = (int(x / 60))
        # Print "Reset in {} min.format(int(x))
        print("Reset in {} min".format(x))
    # status code == 404
    if r.status_code == 404:
        # print response "not found"
        print("Not found")
    # status code == 200 location= r.json()["location"]
    if r.status_code == 200:
        location = r.json()["location"]
        # print location
        print(location)

        # Print "Reset in {} min.format(int(x))
        print("Reset in {} min".format(x))
    # status code == 404
    if r.status_code == 404:
        # print response "not found"
        print("Not found")
    # status code == 200 location= r.json()["location"]
    if r.status_code == 200:
        location = r.json()["location"]
        # print location
        print(location)
