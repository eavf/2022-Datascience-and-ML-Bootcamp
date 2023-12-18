from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
df_data = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                                      'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B','LSTAT'])
features = df_data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIM_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

zillow_median_price = 583.3
scale_factor = zillow_median_price / np.e**np.median(target)

property_stats = features.mean().values.reshape(1,11)

regr = lr().fit(features, target)
fitted_vals = regr.predict(features)

# Challenge: Calculate the MSE and RMSE using sklearn
MSE = mse(target,fitted_vals)

RMSE = np.sqrt(MSE)

# Funkcia na odhad log ceny domu

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    # Calc Range
    if high_confidence:
        # Do X (95%)
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        # Do Y (65%)
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
        
    return log_estimate, upper_bound, lower_bound, interval


def get_dollar_estimate(rm,
                    ptratio,
                    chas=False,
                    large_range=True):
    
    """
    Estimate the price of the property in Boston.
    rm -- number of rooms in the property.
    ptratio -- number of students per teacher in the classroom for the school in the area.
    chas --True if the property is next to river, False otherwise.
    large_range -- True for 95 % prediction interval, False for a 68 % interval.
    
    """
    if rm < 1 or ptratio < 1:
        print('That is unrealistic. Try again')
        return
        
        
    log_est, upper, lower, conf = get_log_estimate(rm,
                                                    students_per_classroom=ptratio,
                                                    next_to_river=chas,
                                                    high_confidence=large_range)
    # convert to todays dollars
    dollar_est = np.e**log_est * 1000 * scale_factor
    dollar_hi = np.e**upper * 1000 * scale_factor
    dollar_lo = np.e**lower * 1000 * scale_factor

    # Round the dollar values to nearest thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_lo = np.around(dollar_lo, -3)

    print(f'The estimated property value is {rounded_est}')
    print(f'AT {conf}% confidence the valuation range is')
    print(f'USD {rounded_lo} at the lower end to USD {rounded_hi} at the high end.')


