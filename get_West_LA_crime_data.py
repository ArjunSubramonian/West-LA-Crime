import numpy as np
import string

import pandas as pd
from sodapy import Socrata

import h5py

# PULLS STRAIGHT FROM DATASET HOST, VIA API

# Crime code descriptions in the dataset are not consistent
def get_crime_code_descs():
    code_descs = {}
    with open('lapd_crime_codes.txt', 'r', encoding = 'utf-8-sig') as descs:
        for line in descs:
            crime = line.split()
            code_descs[int(crime[0])] = ' '.join(crime[1:])
    return code_descs

def get_types_dates_counts():
    # crimes = {K = type of crime, V = {K = date occurred, V = # of occurrences}}
    crimes = {}
    crimes['ALL CRIME'] = {}

    code_descs = get_crime_code_descs()

    # Access dataset
    client = Socrata("data.lacity.org", None)

    for offset in range(0, 2000000, 50000):
        print('Have retrieved ' + str(offset) + ' data points in total.')

        # Results, returned as JSON from API, converted to Python list of dictionaries by sodapy
        results = client.get("7fvc-faax", limit = 50000, offset = offset)

        for result in results:
            # Get area code
            try:
                area_code = int(result['area_id'].strip())
                # Skip over areas that are not West LA (West Los Angeles Division Area Code = 8)
                if area_code != 8:
                    continue
            except:
                continue

            # Get crime type
            try:
                # crime_code = int(result['crm_cd'].strip())
                # crime_type = code_descs[crime_code]
                crime_type = result['crm_cd_desc'].strip()
            except:
                continue

            # Get date
            # Convert floating timestamp datatype to string
            date_occurred = str(result['date_occ'])
            # Convert date to yyyy/mm format
            date_occurred = '/'.join(date_occurred.split('-')[0:2])

            # Add crime instance to crimes dictionary
            if crime_type not in crimes:
                crimes[crime_type] = {date_occurred : 1}
            else:
                if date_occurred not in crimes[crime_type]:
                    crimes[crime_type][date_occurred] = 1
                else:
                    # Keep a running tally of number of occurrences of type of crime on date
                    crimes[crime_type][date_occurred] += 1

            if date_occurred not in crimes['ALL CRIME']:
                crimes['ALL CRIME'][date_occurred] = 1
            else:
                crimes['ALL CRIME'][date_occurred] += 1

    return crimes

def get_types_ages():
    # crimes = {K = type of crime, V = [ages]}
    crimes = {}
    crimes['ALL CRIME'] = []

    code_descs = get_crime_code_descs()

    # Access dataset
    client = Socrata("data.lacity.org", None)

    for offset in range(0, 2000000, 50000):
        print('Have retrieved ' + str(offset) + ' data points in total.')

        # Results, returned as JSON from API, converted to Python list of dictionaries by sodapy
        results = client.get("7fvc-faax", limit = 50000, offset = offset)

        for result in results:
            # Get area code
            try:
                area_code = int(result['area_id'].strip())
                # Skip over areas that are not West LA (West Los Angeles Division Area Code = 8)
                if area_code != 8:
                    continue
            except:
                continue

            # Get crime type
            try:
                # crime_code = int(result['crm_cd'].strip())
                # crime_type = code_descs[crime_code]
                crime_type = result['crm_cd_desc'].strip()
            except:
                continue

            # Get victim's age
            try:
                victim_age = int(result['vict_age'].strip())
            except:
                continue

            # Add crime instance to crimes dictionary
            if crime_type not in crimes:
                crimes[crime_type] = [victim_age]
            else:
                crimes[crime_type].append(victim_age)
            crimes['ALL CRIME'].append(victim_age)

    return crimes

def get_crime_code_indices():
    code_indices = {}
    counter = 0
    with open('lapd_common_crimes.txt', 'r', encoding = 'utf-8-sig') as codes:
        for line in codes:
            crime = line.split()
            code_indices[int(crime[0])] = counter
            counter += 1
    return code_indices

def get_features():
    simplify_classes = {
        110 : 1,
        113 : 1,
        121 : 2,
        122 : 2,
        230 : 3,
        250 : 4,
        330 : 5,
        331 : 5,
        350 : 6,
        351 : 6,
        352 : 6,
        353 : 6,
        410 : 5,
        420 : 5,
        421 : 5,
        440 : 6,
        441 : 6,
        450 : 6,
        451 : 6,
        452 : 6,
        480 : 7,
        485 : 7,
        510 : 7,
        520 : 7,
        624 : 3,
        625 : 3,
        647 : 4,
        753 : 1,
        756 : 1,
        761 : 1,
        762 : 8,
        763 : 9,
        815 : 3,
        845 : 3,
        850 : 8,
        860 : 3,
        910 : 10,
        920 : 10,
        922 : 10,
        930 : 11,
        940 : 11
    }

    # features: month, time of day, victim age, victim sex (0 = female, 1 = male, 2 = unknown), descent (position in alphabet), location (x, y))
    X_train = []
    Y_train = []

    code_indices = get_crime_code_indices()

    # Access dataset
    client = Socrata("data.lacity.org", None)

    counter = 0

    for offset in range(0, 2000000, 50000):
        print('Have retrieved ' + str(offset) + ' data points in total.')

        # Results, returned as JSON from API, converted to Python list of dictionaries by sodapy
        results = client.get("7fvc-faax", limit = 50000, offset = offset)

        for result in results:
            # Get area code
            try:
                area_code = int(result['area_id'].strip())
                # Skip over areas that are not West LA (West Los Angeles Division Area Code = 8)
                if area_code != 8:
                    continue
            except:
                continue

            example = []
            try:
                month = int(str(result['date_occ']).strip().split('-')[1])
                time = int(result['time_occ'].strip())
                age = int(result['vict_age'].strip())

                sex = result['vict_sex'].strip()
                if sex == 'F':
                    sex = 0
                elif sex == 'M':
                    sex = 1
                elif sex == 'X':
                    sex = 2
                else:
                    continue

                descent = result['vict_descent'].strip()
                descent = string.ascii_uppercase.index(descent) + 1

                loc = str(result['location_1']['coordinates']).split(',')
                loc_x = float(loc[0].strip()[1:])
                loc_y = float(loc[1].strip()[:-1])

                example = [month, time, age, sex, descent, loc_x, loc_y]
            except:
                continue

            example = np.array(example)[np.newaxis]
            example = example.T

            try:
                # Get crime type
                crime_code = int(result['crm_cd'].strip())
                index = code_indices[simplify_classes[crime_code]]
            except:
                continue

            if counter == 0:
                X_train = example
                counter += 1
            else:
                X_train = np.append(X_train, example, 1)
            Y_train.append(index)

    with h5py.File("train.h5", "w") as f:
        f.create_dataset('data_X', data=X_train)
        f.create_dataset('data_Y', data=np.array(Y_train))
