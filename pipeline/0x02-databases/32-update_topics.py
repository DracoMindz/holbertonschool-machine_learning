#!/usr/bin/env python3
"""
function that changes all topics of a school doc
based on the name
"""


def update_topics(mongo_collection, name, topics):
    """ changes topics based on name"""
    documents = mongo_collection.update_many({'name': name},
                                             {'$set': {'topics': topics}})
