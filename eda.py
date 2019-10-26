import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# load data and take a look at it
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/train.csv")
df_all = [df_train, df_test]


def replace_nan(df, cols, strategy, value=None, group=None):

	for col in cols:
		if strategy == 'mean':
			val = df.loc[df[col].isnull() == False, col].mean()
			df[col].fillna(val, inplace=True)
		elif strategy == 'median':
			val = df.loc[df[col].isnull() == False, col].median()
			df[col].fillna(val, inplace=True)
		elif strategy == 'mode':
			val = df.loc[df[col].isnull() == False, col].mode()
			df[col].fillna(val, inplace=True)
		elif strategy == 'const':
			val = value
			df[col].fillna(val, inplace=True)
		elif strategy == 'mean_group':
			df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
		elif strategy == 'median_group':
			df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
		elif strategy == 'mode_group':
			df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.mode()))


mean_group_cols = ['LotFrontage', 'MasVnrArea']
grp = 'Neighborhood'
for df in df_all:
	replace_nan(df, mean_cols, 'mean_group', group=grp)

mode_cols = ['Electrical']
for df in df_all:
	replace_nan(df, mode_cols, 'mode')

none_cols = ['Alley', 'MasVnrType' 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
			 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
			 'Fence', 'MiscFeature']
for df in df_all:
	replace_nan(df, none_cols, 'const', value='None')