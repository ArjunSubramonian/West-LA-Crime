import matplotlib.pyplot as plt
import numpy as np

import get_West_LA_crime_data as gcd

# Age groups most affected by crime in West LA

# Ways to visualize:
    # By type of crime (vandalism, theft, etc.)
    # Histogram with the following bins:
        # 0-18
        # 19-30
        # 31-50
        # 50+
    # Relative frequency

print('Please wait while the program runs!')

# crimes = {K = type of crime, V = [ages]}
crimes = gcd.get_types_ages()

for crime_type in crimes:
    print(crime_type)
    data = crimes[crime_type]

    fig = plt.figure()

    # Plot
    # weights make plot of relative frequency
    right_edge = max(51, max(data))
    hist = np.histogram(data, bins = [0, 19, 31, 51, right_edge], weights = np.zeros_like(data) + 1. / len(data))
    plt.bar(range(4), hist[0], width = 1, edgecolor = 'black', zorder = 3)

    plt.xlabel('Age Group (yrs)', fontsize = 8)
    plt.ylabel('Relative Frequency', fontsize = 8)
    plt.title('Age Groups of ' + crime_type + ' Victims in West LA', fontsize = 8)

    x = ['0-18', '19-30', '31-50', '50+']
    plt.xticks(range(4), x, fontsize = 8)
    # y scale should always be 0 to 1, with step size 0.1
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 8)

    # Add grid to plot (pushed to back)
    plt.grid(True, zorder = 0)
    fig.tight_layout()

    crime_type = crime_type.replace('/', '_')
    # Save the plot
    plt.savefig('./Victims_Age_Groups/' + crime_type + '.png')