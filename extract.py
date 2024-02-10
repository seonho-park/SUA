from typing import Any
import pandas as pd
import numpy as np
from scipy.stats import burr12
from collections import defaultdict

INPUT_FILE_NAME = 'Input_SUA.xlsx'

def load_input_file() -> tuple[dict[int,Any], dict[int,Any]]:
    demand_pd = pd.read_excel(INPUT_FILE_NAME, sheet_name = 'Demand')
    variance_pd = pd.read_excel(INPUT_FILE_NAME, sheet_name = 'Demand Variance')

    # construct product data
    product = {}
    num_product = demand_pd.shape[1] - 1
    for i in range(num_product):
        data = demand_pd[i].to_numpy()
        product[i] = {
            'demand': data[0],
            'variance_id': int(data[1]),
            'margin': data[2],
            'cogs': data[3],
            'capacity': data[4],
            # 'price': data[2]+data[3], # margin + cogs
            'substitutability_id': int(data[5])
        }
    
    # construct distributions
    num_dist = variance_pd.shape[0]
    dist = {}
    for i in range(num_dist):
        data = variance_pd.loc[i]
        dist[i] = burr12(c = data['c'], d = data['d'], loc = data['loc'], scale = data['scale'])

    return product, dist