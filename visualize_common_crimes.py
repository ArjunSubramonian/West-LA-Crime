import matplotlib.pyplot as plt
import numpy as np

import get_West_LA_crime_data as gcd

# Common crimes in the last 10 years in West LA

# Ways to visualize:
    # By type of crime (vandalism, theft, etc.)
    # Relative frequency bar graph
        # Only include bars for crimes above 5%
        # Use an "OTHER" bar for the remaining crimes
    
print('Please wait while the program runs!')

# crimes = {K = type of crime, V = {K = date occurred, V = # of occurrences}}
crimes = gcd.get_types_dates_counts()

# Keep running tally of total number of crime occurrences in West LA over the last 10 years
num_total_crimes = 0
for date in crimes['ALL CRIME']:
    num_total_crimes += crimes['ALL CRIME'][date]

# Keep running tallies of total number of occurrences of each type of crime in West LA over the last 10 years
# Group types of crimes with frequency less than 5% into "OTHER" category
bars = {}
other = 0
for crime_type in crimes:
    if crime_type == 'ALL CRIME':
        continue

    total = 0
    for date in crimes[crime_type]:
        total += crimes[crime_type][date]
    if total / num_total_crimes >= 0.05:
        # Don't want overly long x-tick labels
        crime_type = crime_type.split('(')[0].strip()

        bars[crime_type] = total / num_total_crimes
    else:
        other += total / num_total_crimes

# + 1 because of "OTHER" category
ind = np.arange(len(bars) + 1)    # the x locations for the groups

x = sorted(list(bars.keys()))
heights = [bars[crime_type] for crime_type in x]

x.append('OTHER')
heights.append(other)

fig = plt.figure()

# Plot
p = plt.bar(ind, heights, width = 0.35, zorder = 3)

plt.ylabel('Relative Frequency', fontsize = 8)
plt.title('Common Crimes in West LA', fontsize = 8)
plt.xticks(ind, x, rotation = 70, fontsize = 8)
# y scale should always be 0 to 1, with step size 0.1
plt.yticks(fontsize = 8)

# Add grid to plot (pushed to back)
plt.grid(True, zorder = 0)
fig.tight_layout()

# Save the plot
plt.savefig('./Common_Crimes/common_crimes.png')
