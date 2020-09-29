#!/usr/bin/python3
"""
Method that returns the list of ships
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
    url = "https://swapi-api.hbtn.io/api/starships"
    ships = []

    while url is not None:
        r = requests.get(url)
        qr = r.json()
        results = qr['results']

        # get number of passenger on each ship
        for ship in results:
            peps = ship["passengers"]
            peps = peps.replace(",", "")
            if peps.isnumeric() and int(peps) >= passengerCount:
                ships.append(ship["name"])
        url = qr["next"]
    return ships
