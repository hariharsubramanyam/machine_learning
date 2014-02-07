import numpy as np
import matplotlib.pyplot as plt

'''
Sn is a list with elements of the form (x,y) where x is a numpy array and y is -1 or 1
	each x is a feature vector of dimension d
	each y is a label
h is a hypothesis function which takes a d-dimensional vector and maps it to a label 1 or -1

return: the number of misclassifications divided by the number of training examples
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
	each x is a feature vector of dimension d
	each y is a label
max_iterations is the number of maximum number of cycles allowed before we call it quits
through_origin is True if we want our separating boundary to pass through the origin
epsilon is used to determine if something is nearly zero (i.e. x is nearly zero if abs(x) <= epsilon)
verbose is true if we want debugging output

return: (theta, theta_zero) where theta is a d-dimensional numpy array and theta_zero is a number.
This represents the hypothesis:
h(x;theta, theta_zero) = sign(dot(theta,x)+theta_zero)
'''
def perceptron(Sn, max_iterations=500, through_origin=False, verbose=False):
	n = len(Sn)
	if(n == 0):
		raise Exception("Must have some training data!")
	theta_zero = 0
	theta = np.zeros(Sn[0][0].shape)
	current_iteration = 0 			# keep track of the number of iterations
	correct_classified_streak = 0
	while current_iteration < max_iterations and correct_classified_streak < n: # stop if we've exceeded max allowed iterations or if we've converged
		correct_classified_streak += 1
		index = current_iteration%n
		(x,y) = Sn[index]
		if verbose:
			print "On iteration", current_iteration, "with streak", correct_classified_streak, "considering sample", index
			print "Sample label:", y, "and features:", x
		if y*(np.dot(theta,x) + theta_zero) <= 0:	# have we misclassified this training example?
			if verbose:
				print "misclassified sample"
				print "old theta", theta, theta_zero
			correct_classified_streak = 0
			has_some_misclassifications = True
			theta = theta + y*x 				# theta update rule
			if not through_origin: 					# if we want a decision boundary through the origin, we cannot update theta_zero
				theta_zero = theta_zero + y 	# theta_zero update rule
			if verbose:
				print "new theta", theta, theta_zero
		current_iteration += 1
	if verbose:
		print "completed after", current_iteration, "iterations"
		print "theta =", theta, "theta_zero=",theta_zero
		print
		print
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
		Y = [(-1.0*theta_zero - theta[0]*X[0])/theta[1], (-1.0*theta_zero - theta[0]*X[1])/theta[1]]
		plt.plot(X, Y, color="k", linestyle="-",linewidth=2)
	plt.show()


Sn = [
(np.array([-1,-3]), 1),
(np.array([3,4]), 1),
(np.array([-2,-2]), -1),
(np.array([-1,-2]), -1)
]

print plot_2D_training(Sn,perceptron(Sn, verbose=True))