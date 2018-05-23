import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data_train = pd.read_csv('./input/train.csv')
data_test = pd.read_csv('./input/test.csv')

data_train.sample(3)