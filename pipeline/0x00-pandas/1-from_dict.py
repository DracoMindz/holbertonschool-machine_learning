#!/usr/bin/env python3
"""
Function creates a pd.DataFrame from a dictionary
"""

import numpy as np
import pandas as pd


dataDict = {'First': [0.0, 0.5, 1.0, 1.5],
            'Second': ['one', 'two', 'three', 'four']}
# name the columns using index
df = pd.DataFrame(dataDict, index=list('ABCD'))
