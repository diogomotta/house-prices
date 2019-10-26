import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def replace_nan(df, cols, strategy, value=None, group=None):
	for col in cols:
		if strategy == 'mean':
			val = df.loc[df[col].isnull() == False, col].mean()
			df[col].fillna(val, inplace=True)
		elif strategy == 'median':
			val = df.loc[df[col].isnull() == False, col].median()
			df[col].fillna(val, inplace=True)
		elif strategy == 'mode':
			val = df.loc[df[col].isnull() == False, col].mode()[0]
			df[col].fillna(val, inplace=True)
		elif strategy == 'const':
			val = value
			df[col].fillna(val, inplace=True)
		elif strategy == 'mean_group':
			df[col] = df.groupby([group])[col].transform(lambda x: x.fillna(x.mean()))
		elif strategy == 'median_group':
			df[col] = df.groupby([group])[col].transform(lambda x: x.fillna(x.median()))
		elif strategy == 'mode_group':
			df[col] = df.groupby([group])[col].apply(lambda x: x.fillna(x.mode()[0]))

def encode_labels(df, cols):
	for col in cols:
		label = LabelEncoder() 
		label.fit(list(df[col].values)) 
		df[col] = label.transform(list(df[col].values))

def numerical_to_categorical(df, cols):
	for col in cols:
		df[col] = df[col].astype(str)