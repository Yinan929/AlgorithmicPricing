import pandas as pd
import numpy as np
import pathlib as path
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import datetime
import random
import numdifftools as nd
from scipy.special import logsumexp
from knitro import *
from knitro.numpy import *
from knitro.scipy import kn_minimize
# from common_import import raw_dir, data_dir,tab_dir,fig_dir,write_tex_table
from scipy.stats import zscore
from scipy.stats import qmc
from scipy.stats import norm

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

#Perform Demand Estimation within each competitive set
#High dimension fixed effect

data_dir = path.Path('/home/yw1633')

CompsetId = int(sys.argv[1])

# CompsetId = 13
N_cluster = 15

seed_value = 42
random.seed(seed_value)

#Define a class for Demand Estimation
class EstimatingDemand(object):
	#Constructor1
	def __init__(self,df):
		#observations converted to recarray
		self.df=df
		self.df['mkt_id'] = self.df.groupby(['check_in','Lead_Time']).ngroup()

		#Number of observations
		self.N = self.df.shape[0]
		#number of markets
		self.unique_market_ids = np.unique(self.df.mkt_id)
		self._market_indices = {t: i for i, t in enumerate(self.unique_market_ids)}
		self._product_market_indices = self.get_indices(self.df.mkt_id)
		self.T = self.unique_market_ids.size

		#number of properties
		self.unique_prop_ids = np.unique(self.df.Property_ID)
		self.num_pps = len(self.unique_prop_ids)
		self._prop_indices = {t: i for i, t in enumerate(self.unique_prop_ids)}

		#Design Matrix
		self.prices = self.extract_matrix('Price')
		self.Res = self.extract_matrix('Reserved_Flag')
		self.Search = self.extract_matrix('Search')
		self.hotel = self.extract_matrix('Occupancy')
		self.pp_id = self.extract_matrix('Property_ID')

		self.D = np.c_[self.df['Lead_Time'],
				self.df['Is_Weekend'],self.df['Is_Holiday'],self.df['Near_Holiday'],\
				self.df['Jan'],self.df['Feb'],self.df['Mar'],self.df['Apr'],self.df['May'],self.df['Jun'],\
				self.df['July'],self.df['Aug'],self.df['Sep'],self.df['Oct'],self.df['Nov'],self.df['Dec']]



	def get_indices(self,ids):
		"""From a one-dimensional array input, construct a dictionary with keys that are the unique values of the array
		and values that are the indices where the key appears in the array.
		"""
		flat = np.array(ids)
		sort_indices = flat.argsort(kind='mergesort')
		sorted_ids = flat[sort_indices]
		changes = np.ones(flat.shape, np.bool_)
		changes[1:] = sorted_ids[1:] != sorted_ids[:-1]
		reduce_indices = np.nonzero(changes)[0]
		return dict(zip(sorted_ids[reduce_indices], np.split(sort_indices, reduce_indices)[1:]))

	def extract_matrix(self, key):
		"""Attempt to extract a field from a structured array-like object or horizontally stack field0, field1, and so on,
		into a full matrix. The extracted array will have at least two dimensions.
		"""
		matrix = np.c_[self.df[key]]
		return matrix if matrix.size > 0 else None



	def compute_delta(self,theta,pp_fe,s):
		mkt_s = self._product_market_indices[s]
		J = mkt_s.size
		
		#coefficient for price
		alpha = theta[0]
		beta = theta[1]
		gamma = theta[2:]

		propery_id_in = self.pp_id[mkt_s].flatten()
		ppid_idx = np.array([self._prop_indices[key] for key in propery_id_in])
		fixed_effect = np.array(pp_fe)[ppid_idx]

		delta = np.c_[fixed_effect] + alpha*self.prices[mkt_s] + beta*self.hotel[mkt_s] + self.D[mkt_s] @ np.c_[gamma]

		return delta

	def compute_lamda(self,theta2,s):
		mkt_s = self._product_market_indices[s] 		
		lamda = 10*np.exp(self.D[mkt_s] @ np.c_[theta2[:-1]] + theta2[-1]*self.Search[mkt_s])
		return lamda
		


	def log_sjmt(self,theta,pp_fe,s):
		delta = self.compute_delta(theta,pp_fe,s)
		denom = np.concatenate((np.array([[0]]), delta),axis=0)
		log_sum = logsumexp(denom,axis=0)
		
		return delta - log_sum


	def der_Sjtm(self,s,theta1,pp_fe):
		mkt_s = self._product_market_indices[s]
		J = mkt_s.size
		s_jmt = np.exp(self.log_sjmt(theta1,pp_fe,s))
		X = np.c_[self.prices[mkt_s],self.hotel[mkt_s],self.D[mkt_s]]
		mkt_sum_der = (X*s_jmt).sum(axis=0) # J x 20 - > 1 x 20
		der = X - mkt_sum_der

		return  der

	def der_Lam(self,s,theta2):
		'''
		'''
		mkt_s = self._product_market_indices[s]
		lamda = self.compute_lamda(theta2,s)
		X = np.c_[self.D[mkt_s],self.Search[mkt_s]]
		d = X*lamda

		return d

	def der_Fe(self,theta1,pp_fe,s,lamd,A):
		d = np.zeros(self.num_pps)
		logs_jmt = self.log_sjmt(theta1,pp_fe,s) #JX1
		s_jmt = np.exp(logs_jmt)
		propery_id_in = self.pp_id[self._product_market_indices[s]].flatten()
		ppid_idx = np.array([self._prop_indices[key] for key in propery_id_in])

		t = np.ones(logs_jmt.shape[0])*sys.float_info.epsilon
		sub = np.maximum(np.exp(lamd*s_jmt), 1+t[:, np.newaxis])

		mkt_sum_1 = (A*(-lamd)*s_jmt).sum()
		# mkt_sum_2 = ((1-A)*s_jmt*lamd/(np.exp(lamd*s_jmt)-1)).sum()
		mkt_sum_2 = ((1-A)*s_jmt*lamd/(sub-1)).sum()
		# d[ppid_idx] = (-A*s_jmt*lamd-s_jmt*mkt_sum_1 + (1-A)*s_jmt*lamd/(np.exp(lamd*s_jmt)-1)-s_jmt*mkt_sum_2).flatten()
		d[ppid_idx] = (-A*s_jmt*lamd-s_jmt*mkt_sum_1 + (1-A)*s_jmt*lamd/(sub-1)-s_jmt*mkt_sum_2).flatten()

		return d

	def exp_gradient(self,param):
		grad = 0
		theta1 = param[:18] #15
		theta2 = param[18:35]
		pp_fe = param[35:]
		
		for m in self.unique_market_ids:
			ds = self.der_Sjtm(m,theta1,pp_fe)
			dl = self.der_Lam(m,theta2)
	
			logs_jmt = self.log_sjmt(theta1,pp_fe,m) #JX1
			t = np.ones(logs_jmt.shape[0])*sys.float_info.epsilon
			s_jmt = np.exp(logs_jmt)
			lamd = self.compute_lamda(theta2,m)

			sub = np.maximum(np.exp(lamd*s_jmt), 1+t[:, np.newaxis])

			A = 1 - self.Res[self._product_market_indices[m]]
			# dS = -A*s_jmt*(lamd*ds) + (1-A)*s_jmt*ds*lamd/(np.exp(lamd*s_jmt)-1)
			dS = -A*s_jmt*(lamd*ds) + (1-A)*s_jmt*ds*lamd/(sub-1)
	
			# dlam = -A*s_jmt*dl + (1-A)*s_jmt*dl/(np.exp(lamd*s_jmt)-1)
			dlam = -A*s_jmt*dl + (1-A)*s_jmt*dl/(sub-1)

			dFe = self.der_Fe(theta1,pp_fe,m,lamd[0][0],A)

			grad += np.concatenate(((np.c_[dS,dlam]).sum(axis=0), dFe))
		
		return -grad



	def compute_ll_contribution(self,s,param):
		mkt_s = self._product_market_indices[s]
		J = mkt_s.size
		#For demand process
		theta1 = param[:18] #15
		theta2 = param[18:35]
		pp_fe = param[35:]

		lambd = self.compute_lamda(theta2,s)

		logs_jmt = self.log_sjmt(theta1,pp_fe,s) #JX1
		m = np.ones(logs_jmt.shape[0])*sys.float_info.epsilon
		s_jmt = np.exp(logs_jmt)

		#likelihood
		A = 1 - self.Res[self._product_market_indices[s]]
		sub = np.minimum(np.exp(-lambd*s_jmt),1 - m[:, np.newaxis])

		# log_prob = (A*(-lambd*s_jmt) + (1-A)*np.log(1-np.exp(-lambd*s_jmt))).sum()
		log_prob = (A*(-lambd*s_jmt) + (1-A)*np.log(1-sub)).sum()
		
		return log_prob

	# no parallel
	def log_likelihood(self,param):
		log_prob = 0
		start = timer()

		for m in self.unique_market_ids:
			ll = self.compute_ll_contribution(m,param)
	
			log_prob += ll

		end = timer()
		print(-log_prob)
		print('Total time:', end - start)
		return -log_prob

	def compute_jacobian(self, theta_MLE):
		return nd.Gradient(self.log_likelihood)(theta_MLE)

	def compute_hessian(self,theta_MLE):
		return nd.Hessian(self.log_likelihood)(theta_MLE) 


	def solve_ml(self, theta):
		print("===========================")
		print("start optimize: ", datetime.datetime.now())
		# result = minimize(self.log_likelihood, x0=theta, method=kn_minimize,jac = self.exp_gradient)

		result = minimize(self.log_likelihood, x0=theta, method='L-BFGS-B',jac = self.exp_gradient)
		print("finish optimize: ", datetime.datetime.now())

		return result

	def get_se(self,theta_MLE): #𝐻−1(𝐺𝐺′)𝐻−1
		H = self.compute_hessian(theta_MLE)
		h_inv = np.linalg.inv(-H)
		# G = self.exp_gradient(theta_MLE)[:, np.newaxis]
		# Sigma = G@G.T
		# W_sandwich = h_inv@Sigma@h_inv
		W_sandwich = h_inv

		return np.sqrt(np.diag(W_sandwich))


#Read in data and only keep compset = CompsetId
df = pd.read_parquet(data_dir/'Demand_data.parquet').query(f'comp_set == {CompsetId}')
demand = EstimatingDemand(df)

number_properties = demand.num_pps

# theta1=[-0.00123922,  0.0197644, -0.00337582,
# 	0.0934437,  0.0477988,  0.060125,  0.0565386,
# 	   -0.063319, -0.705283,  0.199717,  0.0171653,
# 	0.033061,  0.001162, -0.031965,  0.013572,0.01,0.01]

# theta2 = [-0.00123922,  0.0197644, -0.0337582,
# 	0.0934437,  0.0477988,  0.060125,  0.0565386,
# 	   -0.063319, -0.0705283,  0.0199717,  0.0171653,
# 	0.033061,  0.001162, -0.031965,  0.013572,0.001]

pp_fe = [random.uniform(0.01, 0.1) for _ in range(demand.num_pps)]

# ll = demand.log_likelihood(theta1+theta2+pp_fe)

# D = demand.exp_gradient(theta1+theta2+pp_fe)
# print(D)

# D2 = demand.compute_jacobian(theta1+theta2+pp_fe)
# print(D2)
initial_guess = pd.read_csv(data_dir/'MLE_result_test.csv')

initial_theta = initial_guess['Parameter'].tolist()[:35]

param = demand.solve_ml(initial_theta+pp_fe)
print(param)


theta_MLE=param.x

d={'Variable':['Price','Hotel','Lead_Time','Is_Weekend',
		'Is_Holiday','Near_Holiday','Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec',
		'Lead_Time','Is_Weekend',
		'Is_Holiday','Near_Holiday','Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec','Search']+list(demand.unique_prop_ids),'Parameter':theta_MLE}
rslt = pd.DataFrame(d)
rslt.to_csv(data_dir/f'MLE_result_full_{CompsetId}.csv')

f_out = data_dir/f'MLE_result_full_{CompsetId}.tex'

tex = rslt.to_latex(index=False)

with open(f_out, "w") as text_file:
	print(tex, file=text_file)
