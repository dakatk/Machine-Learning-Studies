#!/usr/bin/env python3

from json import loads


def get_data(json_file, ids):
    """Load given 'ids' from segmented json objects in 'json_file'"""

    values = dict()

    with open(json_file, 'r') as f:
        
        for (i, line) in enumerate(f):

            json_data = loads(line)
            str_id = str(i + 1)
            
            values[str_id] = dict()

            for jid in ids:
                values[str_id][jid] = json_data[jid]
                
    return values

# All review data can be connected by 'user_id' and 'business_id'
# to chain information from all files together

def get_restaurants_data():
    """Load data of interest from 'restaurants.json' file"""
    
    return get_data('restaurants.json', ['price', 'stars', 'review_count', 'latitude', 'business_id'])


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
