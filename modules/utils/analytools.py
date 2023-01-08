import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns


'''Return outliers
'''
def remove_outliers(
    data,
    exemption_cols=[],
    lower_quantile=0.25,
    upper_quantile=0.75,
    whisker_width = 1.5
):
    data_clean = data.copy()
    
    for col in num_cols:
        # skip exemption cols
        if col in exemption_cols:
            continue
        
        q1, q3 = data[col].quantile(lower_quantile), data[col].quantile(upper_quantile)
        iqr = q3 - q1

        lower_whisker = q1 - whisker_width * iqr
        upper_whisker = q3 + whisker_width * iqr

        data_clean[col] = np.where(
            data[col] > upper_whisker, 
            upper_whisker, 
            np.where(data[col] < lower_whisker, lower_whisker, data[col])
        )
        
    return data_clean
