import matplotlib.pyplot as plt
import pandas
import math
import scipy.stats as stats

import get_West_LA_crime_data as gcd

# Crime occurrences in the last 10 years in West LA

# Ways to visualize:
    # By type of crime (vandalism, theft, etc.)
    # Over time
    # One data point per month
    # Add plot of exponentially-weighted moving average to help see trend in crime occurrences over time
    # Add plot of regression line to help see overall correlation of data
        # Not always appropriate because not every correlation is linear
        # So use with caution!!
    

print('Please wait while the program runs!')

# crimes = {K = type of crime, V = {K = date occurred, V = # of occurrences}}
crimes = gcd.get_types_dates_counts()

# Time series plot for each type of crime AND overall crime in West LA
for crime_type in crimes:
    # Need at least 15 data points to notice a trend
    if len(crimes[crime_type]) < 15:
        continue

    print(crime_type)
    
    fig = plt.figure()
    # Plot title
    plt.title('Time Series Plot of Number of Occurrences of ' + crime_type, fontsize = 8)

    # Extract x-axis values (i.e. dates occurred)
    x = list(crimes[crime_type].keys())
    # Sort dates
    x.sort()

    # Get y-axis values corresponding to sorted x-axis values (i.e. number of occurrences of type of crime)
    y = []
    for date in x:
        y.append(crimes[crime_type][date])

    plt.scatter(range(len(x)), y, label = 'Crime Data', zorder = 3)

    # Axis labels
    plt.xlabel('Time', fontsize = 8)
    plt.ylabel('Number of Occurrences', fontsize = 8)

    # Ticks and tick labels (DON'T want to label every tick)
    x_ind = []
    x_val = []
    for i in range(len(x)):
        if i % int(len(x) / 10) == 0:
            x_ind.append(i)

            # Go from yyyy/mm to mm/yyyy
            x_val_parts = x[i].split('/')
            x_val.append(x_val_parts[1] + '/' + x_val_parts[0])

    plt.xticks(x_ind, x_val, size = 'small', rotation = 70, fontsize = 8)

    step = int((math.ceil(max(y)) + 1 - min(y)) / 5)
    if step == 0:
        step = 1
    # Want at least 5 ticks (unless not possible to have 5 ticks)
    yint = range(min(y), math.ceil(max(y)) + 1, step)
    plt.yticks(yint, size = 'small', fontsize = 8)

    # alpha is rate of decay
    alpha = 0.075
    com = (1 - alpha) / alpha
    # Calculate exponentially-weighted moving average of y-axis values
    # EWMA makes it easy to see trend in scatter and isn't affected significantly by highly unstable data
    y_ewma = pandas.ewma(pandas.Series(y), com = com)

    # Don't want EWMA plot to start from first point
    plt.plot(y_ewma[1:], 'r--', label = 'EWMA', zorder = 3)

    # Plot regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(x)), y)
    x_line = [0, len(x) - 1]
    y_line = [slope * j + intercept for j in x_line]

    plt.plot(x_line, y_line, '-', color = 'purple', label = 'Regression Line (r = ' + str(round(r_value, 2)) + ')', zorder = 3)

    # Add grid to plot (pushed to back)
    plt.grid(True, zorder = 0)
    plt.legend()
    fig.tight_layout()

    crime_type = crime_type.replace('/', '_')
    # Save the plot
    plt.savefig('./Crime_Over_Time/' + crime_type + '.png')
