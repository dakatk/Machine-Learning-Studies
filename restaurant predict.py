## Correlations:
## 1. restaurants.json:
##      - correlate stars/location or stars/price
## 2. reviews.json:
##      - correlate stars/length of review text

import restaurant_reviews_nn as nns


def denormalize(norm, data_max, data_min):
    """Denormalize 1D array of normalized data"""

    return (norm * (data_max - data_min)) + data_min


def correlation_1():

    stars_loc_model, max_stars, min_stars, max_lats, min_lats, max_longs, min_longs = nns.stars_given_latitude(20)

    prediction_lats = nns.normalize([37.8755, 37.87, 37.865], data_max=max_lats, data_min=min_lats)

    for loc in prediction_lats:

        stars = stars_loc_model.predict([loc])[0][0]

        n_loc = denormalize(loc, max_lats, min_lats)
        n_stars = denormalize(stars, max_stars, min_stars)
        
        print(f'For latitude {n_loc}, predicted number of stars is: {n_stars}')


def correlation_2():
    
    len_stars_model, max_lens, min_lens, max_stars, min_stars = nns.length_given_stars(100)

    prediction_stars = nns.normalize([1, 2, 3, 4, 5], data_max=max_stars, data_min=min_stars)

    for stars in prediction_stars:

        length = len_stars_model.predict([stars])[0][0]

        n_stars = denormalize(stars, max_stars, min_stars)
        n_length = denormalize(length, max_lens, min_lens)
        
        print(f'For stars {n_stars} of a given review, predicted length is: {n_length}')
        

if __name__ == '__main__':

    # correlation_1()
    correlation_2()

