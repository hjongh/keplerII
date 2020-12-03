# classifies KOI as exoplanets based on stellar effective temperature and transit depth
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class keplerii:
	def __init__(self):
		# EVERYTHING WE'RE DOING IS IN HERE
		data = self.load_data()

		# just looking at the data
		# here we should split the data into the CONFIRMED and NOT data, and clean it
		self.cx_train, self.cx_test, self.cy_train, self.cy_test, self.fx_train, self.fx_test, self.fy_train, self.fy_test = self.splitCleanedData(data)

		parameter_x = 'koi_steff'
		parameter_y = 'koi_depth'

		self.plot_parameters(self.cx_train, self.cy_train, self.fx_train, self.fy_train)

		self.K_VALUE = 550 # we have calculated this to be our optimal value
		#checkData()

		# this right here finds the optimal k volue
		k_values = [1, 2, 3, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500,
			550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
		errors = [0.644, 0.676, 0.663, 0.668, 0.688, 0.691, 0.686, 0.697, 0.699, 0.703, 0.701, 0.704, 0.702, 0.705, 0.710, 0.712, 0.712, 0.715, 0.713,
			0.709, 0.713, 0.709, 0.710, 0.709, 0.704, 0.705, 0.702, 0.702, 0.701, 0.700, 0.699, 0.699]
		#kErrors = kErrorFunction(k_values)
		#graphError(k_values, kErrors)

		# now, to predict stars, just run this command!
		self.testModel(self.cx_test, self.cy_test, self.fx_test, self.fy_test, self.K_VALUE)

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
			'koi_time0bk', 'koi_time0bk_err1','koi_time0bk_err2',
			'koi_impact', 'koi_impact_err1','koi_impact_err2',
			'koi_duration', 'koi_duration_err1','koi_duration_err2',
			'koi_depth', 'koi_depth_err1','koi_depth_err2',
			'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
	    	'koi_teq', 'koi_teq_err1', 'koi_teq_err2',
	    	'koi_insol','koi_insol_err1', 'koi_insol_err2',

	    	'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname',

	    	'koi_steff','koi_steff_err1', 'koi_steff_err2',
	    	'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
	    	'koi_srad', 'koi_srad_err1', 'koi_srad_err2',

	    	'ra', 'dec', 'koi_kepmag']

		data = rawData.values
		return data

	def splitCleanedData(self, data):
		# pulling the necessary fields and splitting into confirmed and false
		ksconfirmed = [row[38] for row in data if row[4] == 'CONFIRMED']
		kdconfirmed = [row[23] for row in data if row[4] == 'CONFIRMED']

		ksfalse = [row[38] for row in data if row[4] == 'FALSE POSITIVE']
		kdfalse = [row[23] for row in data if row[4] == 'FALSE POSITIVE']

		# removing invalid planets from confirmed
		toRemoveC = []
		for i in range(len(ksconfirmed)):
			if ksconfirmed[i] != ksconfirmed[i] or kdconfirmed[i] != kdconfirmed[i]:
				toRemoveC.append(i)
		ksconfirmed = np.delete(ksconfirmed, toRemoveC, axis = 0)
		kdconfirmed = np.delete(kdconfirmed, toRemoveC, axis = 0)
		print "Removed %s confirmed planets without necessary fields" % len(toRemoveC) # removes 1 here

		# removing invalid planets from false positive
		toRemoveF = []
		for i in range(len(ksfalse)):
			if ksfalse[i] != ksfalse[i] or kdfalse[i] != kdfalse[i] or kdfalse[i] > 1400000: # removed 1 huge outlier here
				toRemoveF.append(i)
		ksfalse = np.delete(ksfalse, toRemoveF, axis = 0)
		kdfalse = np.delete(kdfalse, toRemoveF, axis = 0)
		print "Removed %s false planets without necessary fields" % len(toRemoveF) # removes 299 here

		cx_train, cx_test, cy_train, cy_test = train_test_split(ksconfirmed, kdconfirmed, test_size = 0.2, train_size = 0.8, random_state = 16)
		fx_train, fx_test, fy_train, fy_test = train_test_split(ksfalse, kdfalse, test_size = 0.2, train_size = 0.8, random_state = 16)

		return cx_train, cx_test, cy_train, cy_test, fx_train, fx_test, fy_train, fy_test

	def plot_parameters(self, cx_train, cy_train, fx_train, fy_train):
		# for confirmed and false positive: steff vs depth
		# steff is the photospheric temperature of the star
		# depth gives the fraction of stellar flux lost at the minimum of planetary transit

		plt.plot(fx_train, fy_train, 'rs', markersize = 2, label = 'False Positives')
		plt.plot(cx_train, cy_train, 'bs', markersize = 2, label = 'Confirmed')
		plt.title('Stellar Effective Temp vs Transit Depth')
		plt.legend()
		plt.xlabel('Stellar Effective Temperature (Kelvin)')
		plt.ylabel('Transit Depth (ppm)')
		plt.show()

	def calculateSortedDistances(self, steff, depth, cx, fx, cy, fy):
		distances = []
		for i in range(len(cx)):
			distance = np.sqrt( (steff - cx[i])**2 + (depth - cy[i])**2 )
			distances.append([distance,'CONFIRMED'])
		for i in range(len(fx)):
			distance = np.sqrt( (steff - fx[i])**2 + (depth - fy[i])**2 )
			distances.append([distance,'FALSE'])
		distances = sorted(distances)

		return distances

	def returnClassification(self, distances, k):
		CONFIRMED_COUNT = 0
		FALSE_COUNT = 0
		for i in range(k):
			if distances[i][1] == 'CONFIRMED':
				CONFIRMED_COUNT += 1
			else:
				FALSE_COUNT += 1

		if CONFIRMED_COUNT > FALSE_COUNT:
			return 'CONFIRMED'
		else:
			return 'FALSE'

	def testModel(self, cxt, cyt, fxt, fyt, k):
		correct = 0
		total = len(cxt) + len(fxt)
		for i in range(len(cxt)):
			distances = self.calculateSortedDistances(cxt[i], cyt[i], self.cx_train, self.fx_train, self.cy_train, self.fy_train)
			result = self.returnClassification(distances, k)
			if result == 'CONFIRMED':
				correct += 1

		for i in range(len(fxt)):
			distances = self.calculateSortedDistances(fxt[i], fyt[i], self.cx_train, self.fx_train, self.cy_train, self.fy_train)
			result = self.returnClassification(distances, k)
			if result == 'FALSE':
				correct += 1
		accuracy = correct/float(total)
		print '%s nearest neighbors had accuracy of %.3f' % (k, accuracy)
		return accuracy

	def checkData(self):
		print '%s correct and %s false training entries' % (len(self.cx_train), len(self.fx_train))
		print '%s correct and %s false test entries' % (len(self.cx_test), len(self.fx_test))

	def kErrorFunction(self, k):
		error = []
		for i in k:
			accuracy = self.testModel(self.cx_test, self.cy_test, self.fx_test, self.fy_test, i)
			error.append(accuracy)
		return error

	def graphError(self, k_values, error):
		plt.plot(k_values, error, 'bo', markersize = 4)
		plt.title('Error over K')
		plt.xlim(0, 1005)
		plt.ylabel('Accuracy')
		plt.xlabel('K-value')
		plt.show()

runProgram = keplerii()
