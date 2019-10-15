#!/usr/bin/env python3

from json import loads

import numpy.random as rand


def get_data(json_file, ids, samples=500):
    """Load given 'ids' from segmented json objects in 'json_file'"""

    values = dict()

    rand_count = 0

    with open(json_file, 'r') as f:
        
        for (i, line) in enumerate(f):

            if rand.uniform() <= 0.5:
                continue

            if samples > 0:
                rand_count += 1

            json_data = loads(line)
            str_id = str(i + 1)
            
            values[str_id] = dict()

            for jid in ids:
                values[str_id][jid] = json_data[jid]

            if rand_count > samples:
                break

    return values


# Pull data of interest from various json files:
def get_restaurants_data():
    """Load data of interest from 'restaurants.json' file"""
    
    return get_data('restaurants.json', ['price', 'stars', 'review_count', 'latitude', 'longitude', 'business_id'])


def get_reviews_data():
    """Load data of interest from 'reviews.json' file"""

    return get_data('reviews.json', ['stars', 'text', 'business_id', 'user_id'])


def get_users_data():
    """Load data of interest from 'users.json' file"""

    return get_data('users.json', ['review_count', 'average_stars', 'user_id'])


if __name__ == '__main__':
    
    print(get_restaurants_data())
    print(get_reviews_data())
    print(get_users_data())
