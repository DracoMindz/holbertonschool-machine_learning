#!/usr/bin/env python3
""" Create method that returns home planets of all sentient species"""

import requests


def sentientPlanets():
    """
    create list of home planet names
    :return: list of names
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []  # ships is a list
    while url is not None:  # if url exists
        r = requests.get(url)  # get info
        results = r.json()["results"]  # request data ion ships

        # get number of passenger on each ship
        for spec in results:
            if (spec["designation"] == "sentient"
                    or spec["classification"] == "sentient"):
                plUrl = spec["homeworld"]
                if plUrl is not None:
                    # request info from api
                    p = requests.get(plUrl).json()
                    # add name to list
                    planets.append(p["name"])
        # next specie in the list
        url = r.json()["next"]
    return planets
