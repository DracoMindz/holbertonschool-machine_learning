#!/usr/bin/env python3
"""
function that returns the list ot school
based on topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    return list of schools based on topic
    :param mongo_collection: colection object
    :param topic: topic searched
    :return: list
    """
    schoolList = mongo_collection.find({'topic': {'$in': [topic]}})
    return schoolList
