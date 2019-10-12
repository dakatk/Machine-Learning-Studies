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

    stars_loc_model, max_stars, min_stars, max_lats, min_lats = nns.stars_given_latitude(100)

    prediction_lats = nns.normalize([37.8755, 37.87, 37.865], data_max=max_lats, data_min=min_lats)

    for loc in prediction_lats:

        n_stars = denormalize(stars_loc_model.predict([loc]), mox_stars, min_stars)
        print(f'For latitude {loc}, predicted number of stars is: {n_stars}')
        

def correlation_2():

    stars_len_model, max_stars, min_stars, max_lens, min_lens = nns.stars_given_length(100)

    prediction_lengths = nns.normalize([55, 250, 270, 75, 60, 190, 145, 230], data_max=max_lens, data_min=min_lens)

    for length in prediction_lengths:

        n_stars = denormalize(stars_len_model.predict([length]), max_stars, min_stars)
        print(f'For length {length} of a given review, predicted number of stars is: {n_stars}')


if __name__ == '__main__':

    # correlation_1()
    correlation_2()

