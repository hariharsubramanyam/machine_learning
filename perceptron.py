import numpy as np
import matplotlib.pyplot as plt

'''
Sn is a list of the form (x,y) where x is a numpy array and y is -1 or 1
h is a hypothesis function
'''
def error(h, Sn):
	n = len(Sn)
	total_sum = 0.0
	for (x, y) in Sn:
		if h(x) != y:
			total_sum += 1.0
	return total_sum / n

'''
Sn is a list of the form (x,y) where x is a numpy array and y is -1 or 1
'''
def perceptron(Sn, max_iterations=500, through_origin=False, epsilon=0.01):
	def near_zero(x):
		print x
		for i in x:
			if abs(i) > epsilon:
				return False
		return True
	n = len(Sn)
	if(n == 0):
		raise Exception("Must have some training data!")
	theta_zero = 0
	theta = np.zeros(Sn[0][0].shape)
	delta = np.ones(theta.shape)	# When delta does not change, we are done with perceptron
	current_iteration = 0
	while current_iteration < max_iterations and not near_zero(delta):
		index = current_iteration%n
		(x,y) = Sn[index]
		if y*(np.dot(theta,x) + theta_zero) <= 0:
			new_theta = theta + y*x
			delta = new_theta - theta
			if not through_origin:
				new_theta_zero = theta_zero + y
				if near_zero(delta):
					delta = new_theta_zero - theta_zero
				theta_zero = new_theta_zero
			theta = new_theta
		current_iteration += 1
	return (theta, theta_zero)

def plot_2D_training(Sn, theta_params=None):
	pos_x = [x[0] for (x,y) in Sn if y==1]
	neg_x = [x[0] for (x,y) in Sn if y==-1]
	pos_y = [x[1] for (x,y) in Sn if y==1]
	neg_y = [x[1] for (x,y) in Sn if y==-1]
	plt.plot(pos_x, pos_y, "r+")
	plt.plot(neg_x, neg_y, "bx")
	if theta_params is not None:
		(theta, theta_zero) = theta_params
		X = [min(min(pos_x), min(neg_x)),max(max(pos_x), max(neg_x))]
		Y = [(theta_zero - theta[0]*X[0])/theta[1], (theta_zero - theta[0]*X[1])/theta[1]]
		plt.plot(X, Y, color="k", linestyle="-",linewidth=2)
	plt.show()


Sn = [
(np.array([1,2]), 1),
(np.array([3,4]), 1),
(np.array([-2,-2]), -1),
(np.array([-1,-2]), -1)
]

print 
plot_2D_training(Sn,perceptron(Sn))