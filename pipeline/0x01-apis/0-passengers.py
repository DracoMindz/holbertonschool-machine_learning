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
    urlShips = "https://swapi-api.hbtn.io/api/starships/"
    ships = []  # ships is a list
    if urlShips is not None:  # if url exists
        req = requests.get(urlShips)   # get info
        shipData = req.json()["results"]   # request data ion ships

        # get number of passenger on each ship
        for ship in shipData:
            num_p = ship["passengers"]
            num_p = num_p.replace(",", " ")  # replace the separators
            # Check number passengers to passenger count needed
            if num_p.isnumeric() and int(num_p) >= passengerCount:
                ships.append(ship["name"])  # add ship name to list
        url = req.json()["next"]

    return ships
