## Correlations:
## 1. restaurants.json:
##      - correlate stars/location or stars/price
## 2. reviews.json:
##      - correlate stars/length of review text

import restaurant_reviews_nn as nns


def denormalize(norm, data_max, data_min):
    """Denormalize 1D array of normalized data"""

    return (norm * (data_max - data_min)) + data_min


# Correlation 1:

prediction_lats = []

stars_loc_model, max_stars, min_stars = nns.stars_given_latitude(100)

for loc in prediction_lats:

    n_stars = denormalize(stars_loc_model.predict([loc]), mox_stars, min_stars)
    print(f'For latitude {loc}, predicted number of stars is: {n_stars}')

# Correlation 2:

prediction_lengths = []

stars_len_model, max_stars, min_stars = nns.stars_given_length(100)

for length in prediction_lengths:

    n_stars = denormalize(stars_len_model.predict([length]), max_stars, min_stars)
    print(f'For length {length} of a given review, predicted number of stars is: {n_stars}')

