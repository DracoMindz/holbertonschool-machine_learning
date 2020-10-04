#!/usr/bin/env python3
""" list all documents in a collection"""


def list_all(mongo_collection):
    """ list all documents in a collection"""
    """ return an empty list if no document in the collection"""
    documents = []
    collection = mongo_collection.find()
    for docs in collection:
        documents.append(docs)
    return documents
