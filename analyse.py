# Analyzing the data with sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from math import exp

class analyse:
	def __init__(self):
		# EVERYTHING WE'RE DOING IS IN HERE
		data = self.load_data()

		# here we should clean the data, and split it into the features and classification, and train and test data
		x_train, x_test, y_train, y_test = self.cleanSplitData(data)

		#self.completeTrain(x_train, y_train)
		model = self.train(x_train, x_test, y_train, y_test)
		#self.printCoef(model, x_train)

		#self.imbalanceTest(x_train, y_train)

	def completeTrain(self, x_train, y_train):
		model = LogisticRegression()
		model.fit(x_train, y_train)
		coef = model.coef_
		print coef
		weighted = []
		for i in range(len(coef)):
			weighted.append(coef[i] * x_train[:,i].mean())
		print weighted

	def imbalanceTest(self, x_train, y_train):
		confirmed = 0
		for value in y_train:
			if value == 1:
				confirmed += 1
		print 'Out of %s entries, %s are planets.' % (len(y_train), confirmed)

	def train(self, x_train, x_test, y_train, y_test):
		model = LogisticRegression()
		model.fit(x_train, y_train)
		predictedy = model.predict(x_test)
		cnf_matrix = confusion_matrix(y_test, predictedy)

		print 'Predicted with accuracy of %.3f' % accuracy_score(y_test, predictedy)
		print cnf_matrix

		return model

	def printCoef(self, model, x_train):
		weights = []
		coef = model.coef_
		for i in range(len(coef)):
			weights.append(x_train[0][i] * coef[i])
		print coef
		print weights

	def load_data(self):
		# load kepler data from cumulative.csv
		rawData = pd.read_csv('cumulative.csv')

		features = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition',
	    	'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss',
	    	'koi_fpflag_co', 'koi_fpflag_ec',

			'koi_period', 'koi_period_err1', 'koi_period_err2',
			'koi_time0bk', 'koi_time0bk_err1','koi_time0bk_err2', # this is just a time stamp
			'koi_impact', 'koi_impact_err1','koi_impact_err2',
			'koi_duration', 'koi_duration_err1','koi_duration_err2',
			'koi_depth', 'koi_depth_err1','koi_depth_err2',
			'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
	    	'koi_teq', 'koi_teq_err1', 'koi_teq_err2',
	    	'koi_insol','koi_insol_err1', 'koi_insol_err2',

	    	'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname', # ids

	    	'koi_steff','koi_steff_err1', 'koi_steff_err2',
	    	'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
	    	'koi_srad', 'koi_srad_err1', 'koi_srad_err2',

	    	'ra', 'dec', 'koi_kepmag']

	    # add binary classification column
		rawData['classification'] = (rawData.koi_disposition == 'CONFIRMED').astype(int)

		# we drop the ones that are candidates, because this doesnt help us classify
		rawData = rawData[rawData.koi_disposition != 'CANDIDATE']

	    # get only the columns we need
	    # we are using: period, impact, duration, depth, prad, teq, insol, steff, slogg, srad
	    # 'koi_impact', 'koi_depth', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad'
		columnsToRemove = ['koi_impact', 'koi_depth', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad', 'rowid', 'kepid', 'kepoi_name', 'kepler_name' ,'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss','koi_fpflag_co', 'koi_fpflag_ec','koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1','koi_time0bk_err2', 'koi_impact_err1','koi_impact_err2', 'koi_duration_err1','koi_duration_err2', 'koi_depth_err1','koi_depth_err2', 'koi_prad_err1', 'koi_prad_err2','koi_teq_err1', 'koi_teq_err2','koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_steff_err1', 'koi_steff_err2','koi_slogg_err1', 'koi_slogg_err2', 'koi_srad_err1', 'koi_srad_err2','ra', 'dec', 'koi_kepmag']
		columnsToRemoveAll = ['rowid', 'kepid', 'kepoi_name', 'kepler_name' ,'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss','koi_fpflag_co', 'koi_fpflag_ec','koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1','koi_time0bk_err2', 'koi_impact_err1','koi_impact_err2', 'koi_duration_err1','koi_duration_err2', 'koi_depth_err1','koi_depth_err2', 'koi_prad_err1', 'koi_prad_err2','koi_teq_err1', 'koi_teq_err2','koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_steff_err1', 'koi_steff_err2','koi_slogg_err1', 'koi_slogg_err2', 'koi_srad_err1', 'koi_srad_err2','ra', 'dec', 'koi_kepmag']
		for field in columnsToRemoveAll:
			del rawData[field]
		print 'Removed %s fields' % len(columnsToRemoveAll)

		data = rawData.values
		return data

	def cleanSplitData(self, data):
		toRemove = []
		for row in range(len(data)):
			for column in range(len(data[0])):
				if data[row][column] != data[row][column]:
					toRemove.append(row)
		print 'Starting with %s entries.' % len(data)
		data = np.delete(data, toRemove, axis = 0)
		print 'Removed %s entries without necessary fields' % len(toRemove)
		print '%s entries remaining.' % len(data)

		x = data[:, 0:10]
		y = data[:, 10]

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

		return x_train, x_test, y_train, y_test

runProgram = analyse()
