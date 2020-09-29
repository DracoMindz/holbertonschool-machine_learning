#!/usr/bin/python3
"""
Mathod that returns the list of ships
that can hold # passengers
"""
import requests


def availableShips(passengerCount):
    """
    method that returns which ships can hold # passengers
    :param passengerCount: num of passengers
    Note: Include pagination
    :return: of no ship an empty list, list of ships
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []  # ships is a list
    while url is not None:  # if url exists
        r = requests.get(url)   # get info
        results = r.json()["results"]   # request data ion ships

        # get number of passenger on each ship
        for ship in results:
            p = ship["passengers"]
            p = p.replace(",", "")  # replace the separators
            # Check number passengers to passenger count needed
            if p.isnumeric():
                if int(p) >= passengerCount:
                    ships.append(ship["name"])  # add ship name to list
        url = r.json()["next"]

    return ships
