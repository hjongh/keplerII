# classifying koi as exoplanets using logistic regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import exp

class logreg:
	# learning rate, epochs
	lr = 0.0000000001
	epochs = 20

	# parameters
	stemp = 38
	depth = 23
	srad = 44
	prad = 26
	prd = 11

	def __init__(self):
		# EVERYTHING WE'RE DOING IS IN HERE
		data = self.load_data()

		# just looking at the data
		# here we should split the data into the CONFIRMED and NOT data, and clean it
		self.ctrain, self.ctest, self.ftrain, self.ftest = self.splitCleanedData(data)

		parameter_x = 'koi_steff' # 38
		parameter_y = 'koi_depth' # 23
		parameter_z = 'koi_srad'  # 44
		parameter_m = 'koi_prad' # 26
		parameter_n = 'koi_period' # 11
		#self.checkData(data)

		model = self.train(self.ctrain, self.ftrain)
		print self.testModel(model, self.ctest, self.ftest)

	def load_data(self):
		# load kepler data from cumulative.csv
		rawData = pd.read_csv('cumulative.csv')

		# first 5 planets
		#print rawData.head()
		#print rawData.koi_pdisposition[:100]

		features = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition',
	    	'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss',
	    	'koi_fpflag_co', 'koi_fpflag_ec',

			'koi_period', 'koi_period_err1', 'koi_period_err2',
			'koi_time0bk', 'koi_time0bk_err1','koi_time0bk_err2', # this is jsut a time stamp
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

		data = rawData.values
		return data

	def splitCleanedData(self, data):
		# removing invalid rows here
		toRemove = []
		for i in range(len(data)):
			if (data[i][self.stemp] != data[i][self.stemp] or data[i][self.depth] != data[i][self.depth] or data[i][self.srad] != data[i][self.srad] or
				data[i][self.prad] != data[i][self.prad] or data[i][self.prd] != data[i][self.prd] or data[i][self.depth] > 1400000):
				toRemove.append(i)
		data = np.delete(data, toRemove, axis = 0)
		print "Removed %s objects of interest without necessary fields" % len(toRemove)
		print "%s entries remaining." % len(data)

		# get confirmed rows, and false rows
		# [row[self.stemp],row[self.depth],row[self.srad],row[self.prad],row[self.prd]]
		confirmed = [[row[self.stemp],row[self.depth]] for row in data if row[4] == 'CONFIRMED']
		false = [[row[self.stemp],row[self.depth]] for row in data if row[4] == 'FALSE POSITIVE']

		ctrain, ctest = train_test_split(confirmed, test_size = 0.2, train_size = 0.8, random_state = 16)
		ftrain, ftest = train_test_split(false, test_size = 0.2, train_size = 0.8, random_state = 16)

		return ctrain, ctest, ftrain, ftest

	def train(self, cx, fx):
		coef = [0.0 for i in range(len(cx[0]) + 1)]
		for epoch in range(self.epochs):
			netError = 0
			for row in cx:
				y = self.predict(row, coef)
				error = 1 - y
				netError += error**2
				coef[0] = coef[0] + self.lr * error * y * (1 - y)
				for i in range(len(cx[0])):
					coef[i+1] = coef[i+1] + self.lr * error * y * (1 - y) * row[i] # idk if this works
			for row in fx:
				y = self.predict(row, coef)
				error = 0 - y      # idk if this works either
				netError += error**2
				coef[0] = coef[0] + self.lr * error * y * (1 - y)
				for i in range(len(cx[0])):
					coef[i+1] = coef[i+1] + self.lr * error * y * (1 - y) * row[i]
			#if epoch % 20 == 0:
			print 'Epoch %s has error of %.8f, accuracy of %.8f' % (epoch, netError, self.testModel(coef, self.ctest, self.ftest))
		return coef

	def predict(self, features, coefficients):
		y = coefficients[0]
		for i in range(len(features)):
			y += coefficients[i+1] * features[i]
		return exp(y) / (1.0 + exp(y))

	def testModel(self, model, ctest, ftest):
		totalCount = len(ctest) + len(ftest)
		totalCorrect = 0
		for row in ctest:
			prediction = self.predict(row, model)
			if prediction > 0.5:
				totalCorrect += 1
		for row in ftest:
			prediction = self.predict(row, model)
			if prediction < 0.5:
				totalCorrect += 1

		accuracy = totalCorrect / float(totalCount)
		return accuracy

	def checkData(self, data):
		# pulling the necessary fields and seeing what they look like
		prad = [row[26] for row in data if row[4] == 'CONFIRMED']
		srad = [row[44] for row in data if row[4] == 'CONFIRMED']

		for i in range(5):
			print prad[i]
			print srad[i]

runProgram = logreg()
