import pandas as pd
import numpy as np
import pathlib as path
import sys
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import datetime
import random
import numdifftools as nd
from collections import defaultdict
from scipy.special import logsumexp
from common_import import raw_dir, data_dir,tab_dir,fig_dir,write_tex_table
from scipy.stats import zscore
from scipy.stats import qmc
from scipy.stats import norm

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

#Perform Demand Estimation within each competitive set
#High dimension fixed effect

# data_dir = path.Path('/home/yw1633')

# CompsetId = int(sys.argv[1])

CompsetId = 10
N_cluster = 15

# ab-20291729
# 2019-06-09

# seed_value = 42
# random.seed(seed_value)
#Define a class for Computing Optimal Prices
class ComputePrice(object):
	#Constructor1
	def __init__(self,df,param,mu,percent,check_in_date,property_id):
		#Initalizae a few parameters
		self.theta1 = param[:18] 
		self.theta1[0] = -0.02 
		self.theta2 = param[18:35]
		self.pp_fe = param[35:]                                                             
		self.property_id = property_id
		self.mu = mu #price adjustment cost - a number between 0 and 1
		self.f =percent #percentage commission paid to Airbnb

		#observations converted to recarray
		self.df=df.loc[df['check_in'] == check_in_date] #Only keep records for check_in_date
		self.df['mkt_id'] = self.df['Lead_Time']
		self.unique_market_ids = np.unique(self.df.mkt_id)
		#fill in missing markets incase the given property was booked quickly
		self.df = self.fill_missing_mkts(property_id, check_in_date)
		
		#Number of observations
		self.N = self.df.shape[0]
		#number of markets
		
		self._market_indices = {t: i for i, t in enumerate(self.unique_market_ids)}
		self._product_market_indices = self.get_indices(self.df.mkt_id)
		self.T = self.unique_market_ids.size

		#number of properties
		self.unique_prop_ids = np.unique(self.df.Property_ID)
		self.num_pps = len(self.unique_prop_ids)
		self._prop_indices = {t: i for i, t in enumerate(self.unique_prop_ids)}

		self.Vs = np.zeros(181)
		self.Ps = np.zeros(181)

		#Design Matrix
		self.prices = self.extract_matrix('Price')
		self.Res = self.extract_matrix('Reserved_Flag')
		self.Search = self.extract_matrix('Search')
		self.hotel = self.extract_matrix('Occupancy')
		self.pp_id = self.extract_matrix('Property_ID')


		self.D = np.c_[self.df['Lead_Time'],
				self.df['Is_Weekend'],self.df['Is_Holiday'],self.df['Near_Holiday'],self.df['Jan'],\
				self.df['Feb'],self.df['Mar'],self.df['Apr'],self.df['May'],self.df['Jun'],self.df['July'],\
				self.df['Aug'],self.df['Sep'],self.df['Oct'],self.df['Nov'],self.df['Dec']]

	def fill_missing_mkts(self, property_id, check_in_date):
		"""
		Ensures that the DataFrame has rows for a given property_id across all specified mkt_ids.
		Missing rows are filled with data from the nearest mkt_id present for that property_id.
		"""

		# Step 1: Seperate property specific data and mkt specific data
		market_set = self.unique_market_ids
		df_D = self.df[['book_date','check_in','mkt_id','Is_Weekend','Is_Holiday','Near_Holiday','Year',\
				'Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec','Lead_Time','Search','Occupancy']].drop_duplicates()

		df_P = self.df[['Property_ID','mkt_id','comp_set','book_date','check_in','Price','Available_Flag','Reserved_Flag']]


		# Step 1: Fill missing market IDs for df_P
		prop_df = df_P[df_P['Property_ID'] == property_id]

		# Find existing mkt_ids for this property_id
		existing_mkts = prop_df['mkt_id'].unique()

		# Determine missing mkt_ids from the specified list
		missing_mkts = [mkt for mkt in market_set if mkt not in existing_mkts]

		# For each missing mkt_id, find the nearest existing mkt_id and duplicate its row with the new mkt_id
		for missing_mkt in missing_mkts:
			# Find the nearest mkt_id; this simplistic approach just takes any existing mkt_id
			if len(existing_mkts) > 0:

				nearest_mkt_id = min(existing_mkts, key=lambda x: abs(x - missing_mkt))
				nearest_row = prop_df[prop_df['mkt_id'] == nearest_mkt_id].iloc[0]

				# Create a new row with the missing mkt_id, duplicating data from the nearest mkt_id
				new_row = nearest_row.copy()
				new_row['mkt_id'] = missing_mkt
				df_P = pd.concat([df_P, pd.DataFrame([new_row])], ignore_index=True)
			else:
				print(f"No existing market IDs found for property_id {property_id} to duplicate.")

		return pd.merge(df_P.drop(columns=['book_date']),df_D,on=['mkt_id'],how='left')

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

	def compute_delta(self,s,p,idx):
		mkt_s = self._product_market_indices[s]
		new_p = self.prices[mkt_s]
		new_p[idx] = p
		
		#coefficient for price
		alpha = self.theta1[0]
		beta = self.theta1[1]
		gamma = self.theta1[2:]

		propery_id_in = self.pp_id[mkt_s].flatten()
		ppid_idx = np.array([self._prop_indices[key] for key in propery_id_in])
		fixed_effect = np.array(self.pp_fe)[ppid_idx]

		delta = np.c_[fixed_effect] + alpha*new_p + beta*self.hotel[mkt_s] + self.D[mkt_s] @ np.c_[gamma]

		return delta

	def compute_lamda(self,s):
		mkt_s = self._product_market_indices[s]         
		lamda = 10*np.exp(self.D[mkt_s] @ np.c_[self.theta2[:-1]] + self.theta2[-1]*self.Search[mkt_s])
		return lamda[0]
		
	#return choice probabilities for property j only
	def compute_sjmt(self,s,p):
		mkt_s = self._product_market_indices[0]
		propery_id_ary = self.pp_id[mkt_s]
		idx = np.where(propery_id_ary == self.property_id)
		delta = self.compute_delta(s,p,idx)
		denom = np.concatenate((np.array([[0]]), delta),axis=0)
		log_sum = logsumexp(denom,axis=0)
		logs_jmt = delta - log_sum
		s_jmt = np.exp(logs_jmt)
		return s_jmt[idx]

	def compute_q(self,s,p):
		return 1-np.exp(-self.compute_lamda(s)*self.compute_sjmt(s,p))
	
	def compute_P_star_180(self,p):
		q = self.compute_q(0,p)
		return -(q*p*(1-self.f))[0]


	def product_term(self,n, m, p):
		# Calculate the product term
		if n >= m:
			return 1
		else:
			return np.prod([(1 - self.compute_q(s,p)) for s in range(n, m)])


	def V_t_func(self,p,t):
		q = self.compute_q(t,p)
		sum1 = 0
		sum2 = 0

		for i in range(t+1,181):
			qi = self.compute_q(180-i,p)
			Vi = self.Vs[i]
			sum1 = sum1 + (1-self.mu)**(i-t-1)*self.product_term(t+1,i-1,p)*qi
			sum2 = sum2 + (1-self.mu)**(i-t-1)*self.product_term(t+1,i-1,p)*self.mu*Vi

		return (1-self.f)*(q+(1-self.mu)*(1-q)*sum1)*p+(1-q)*sum2

	def neg_V(self,p,t):
		return -self.V_t_func(p,t)

	def compute_all_price(self,t0):
		bounds = (100, 1000)

		result = minimize_scalar(self.compute_P_star_180, bounds=bounds, method='bounded')
		if result.success:
			print('hello')
			print(result)
			self.Vs[180] = -result.fun
			self.Ps[180] = result.x
		else:
			print("Optimization was not successful at the beginning")
			return

		for t in range(179,t0,-1):
			result = minimize_scalar(self.neg_V, bounds=bounds, args=(t),method='bounded')
			
			if result.success:
				print('hello1')
				print(result)
				self.Vs[t] = -result.fun
				self.Ps[t] = result.x
			else:
				print(f"Optimization was not successful at time {t}")
				return 



	# def find_optimal(self):
	#     bounds = (100, 1000)
	#     result = minimize_scalar(self.compute_P_star_180, bounds=bounds, method='bounded')
	#     return result



df = pd.read_parquet(data_dir/'Demand_data.parquet').query(f'comp_set == {CompsetId}')
mu = 0.5
property_id = 'ab-20291729'
check_in_date = '2019-06-09'
percent = 0.03
rslt = pd.read_csv(data_dir/f'MLE_result_full_{CompsetId}.csv')
param = rslt['Parameter'].tolist()

obj = ComputePrice(df,param,mu,percent,check_in_date,property_id)

# V = obj.find_optimal()
# print(V)

obj.compute_all_price(0)
ps = obj.Ps
print(ps)


#     def compute_dq_dp(self,s,property_id):
#         logs_jmt = self.log_sjmt(s) #JX1
#         s_jmt = np.exp(logs_jmt)

		
#         return self.theta1[0]*s_corresponding_to_id*(1-s_corresponding_to_id)

#     # no parallel
#     def log_likelihood(self,param):
#         log_prob = 0
#         start = timer()

#         for m in self.unique_market_ids:
#             ll = self.compute_ll_contribution(m,param)
	
#             log_prob += ll

#         end = timer()
#         print(-log_prob)
#         print('Total time:', end - start)
#         return -log_prob


		

# def compute_p_star(c, f, mu, q, T, E, V):
#     """
#     Compute p_star based on the given expressions.

#     Parameters:
#     - c: Constant value
#     - f: Value for f
#     - mu: Value for mu
#     - q: List of q values from t to T
#     - T: Total number of time steps
#     - E: Function representing the conditional expectation
#     - V: Function representing the conditional expectation for V

#     Returns:
#     - p_star: Computed value for p_star
#     """
#     p_derivative_A = compute_p_derivative_A(mu, q, T, E)
#     A_t = compute_A_t(f, mu, q, T, E)
#     p_derivative_B = compute_p_derivative_B(mu, q, T, V)
#     B_t = compute_B_t(q, T, mu, V)

#     inverse_p_derivative_A = np.linalg.inv(p_derivative_A)

#     p_star = (c / (1 - f)) - np.dot(inverse_p_derivative_A, (A_t + p_derivative_B))

#     return p_star

# def compute_p_derivative_A(mu, q, T, E):
#     # Define the partial derivative of A with respect to p
#     # This is a placeholder, replace it with the actual expression
#     return np.identity(T)  # Replace with the actual expression

# def compute_A_t(f, mu, q, T):
#     """
#     Compute A_t based on the given expression.

#     Parameters:
#     - f: Value for f
#     - mu: Value for mu
#     - q: List of q values from t to T
#     - T: Total number of time steps

#     Returns:
#     - A_t: Computed value for A_t
#     """
#     A_t = 0
#     one_minus_mu_power = 1

#     for m in range(T, 0, -1):
#         product_term = np.prod(1 - q[t] for t in range(m - 1, T - 1, -1))
#         A_t += one_minus_mu_power * product_term * q[m - 1]
#         one_minus_mu_power *= (1 - mu)

#     A_t *= (1 - f) * (q[T - 1] + (1 - mu) * (1 - q[T - 1]))

#     return A_t

# # Example usage:
# # Define the parameters appropriately

# # q_values is a placeholder, replace it with your actual list of q values
# q_values = np.ones(5)

# # Replace f and mu with your actual values
# result = compute_A_t(f=0.5, mu=0.2, q=q_values, T=5)

# print(result)


# def compute_p_derivative_B(mu, q, T, V):
#     # Define the partial derivative of B with respect to p
#     # This is a placeholder, replace it with the actual expression
#     return np.identity(T)  # Replace with the actual expression

# def compute_B_t(q, T, mu, V):
#     # Define the expression for B_t
#     # This is a placeholder, replace it with the actual expression
#     return np.ones(T)  # Replace with the actual expression

# Example usage:
# Define the functions E and V appropriately based on your requirements

# q_values is a placeholder, replace it with your actual list of q values

