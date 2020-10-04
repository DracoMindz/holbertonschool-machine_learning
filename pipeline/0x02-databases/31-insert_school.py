#!/usr/bin/env python3
"""
function inserts a new document in a collection
based on Kwargs
"""


def insert_school(mongo_collection, **kwargs):
    """ inserts new docs based on kwargs"""
    id_ = mongo_collection.insert_one(kwargs).inserted_id
    return id_
