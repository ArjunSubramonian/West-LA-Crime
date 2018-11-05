# West-LA-Crime
Crime in Los Angeles has been a problem for a long time. In many parts of the city, it is unsafe to walk outside alone at night. UCLA, however, is in West LA, which is widely considered to be much safer than other areas. But exactly how safe is West LA? The Los Angeles Police Department publishes crime data online, updated daily (https://goo.gl/PSTrMP). It contains many different data points such as location, area, occurrence data, reporting date, and more. This repository contains visualizations of crime occurrences in the last 10 years in West LA, the most common types of crime in the area, and which age groups are most affected? (0-18, 19-30, 31-50, 50+). In addition, this repository includes Python code to train a deep learning model (fully-connected neural network) (implemented with Keras) to predict the probabilities of being a victim of different types of crime based on the month, time of day, age, sex, descent, and location.

**1. To visualize crime occurrences in West LA over the last 10 years, run:**
`python visualize_crime_over_time.py`

**2. To visualize common types of crime in West LA, run:**
`python visualize_common_crimes.py`

**3. To visualize the age groups of victims of crime, run:**
`python visualize_victims_age_groups.py`

**4. To train the deep learning model, run:**
`python model.py`
*(Trained neural network parameters will be stored in `model.h5`)*

**Note: The files take a little while to run because they pull data straight from the dataset host, via the Soda API.**
