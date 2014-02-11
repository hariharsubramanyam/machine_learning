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
Sn is a list with elements of the form (x,y) where x is a numpy array and y is -1 or 1
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
def perceptron(Sn, max_iterations=500, through_origin=False, verbose=False, start_with=0, plot_after_misclassification=False, initial_theta=None, initial_theta_zero=None):
	n = len(Sn)
	if(n == 0):
		raise Exception("Must have some training data!")
	theta_zero = 0
	theta = np.zeros(Sn[0][0].shape)
	if initial_theta_zero is not None:
		theta_zero = initial_theta_zero
	if initial_theta is not None:
		theta = initial_theta
	current_iteration = start_with 			# keep track of the number of iterations
	correct_classified_streak = 0	# how many examples have we correctly classified in a row? If we've classified all n of them, we've converged
	while current_iteration < max_iterations and correct_classified_streak < n: # stop if we've exceeded max allowed iterations or if we've converged
		index = current_iteration%n
		(x,y) = Sn[index]
		if verbose:
			print
			print "On iteration", current_iteration, "with", correct_classified_streak, "correct classifications in a row. Now considering sample", (index+1)
			print "Sample label:", y, "and features:", x
		if y*(np.dot(theta,x) + theta_zero) <= 0:	# have we misclassified this training example?
			if verbose:
				print "Misclassified! Changing Theta"
				print "old theta", theta, theta_zero
			correct_classified_streak = 0
			has_some_misclassifications = True
			theta = theta + y*x 				# theta update rule
			if not through_origin: 					# if we want a decision boundary through the origin, we cannot update theta_zero
				theta_zero = theta_zero + y 	# theta_zero update rule
			if verbose:
				print "new theta", theta, theta_zero
			if plot_after_misclassification:
				plot_2D_training(Sn, (theta, theta_zero))
		else:
			correct_classified_streak += 1
			if verbose:
				print "Correctly classified!"
		current_iteration += 1
	if verbose:
		print
		print
		print "completed after", current_iteration-1, "iterations"
		print "theta =", theta, "theta_zero =",theta_zero
		print
		print
	return (theta, theta_zero)


'''
Sn is a list with elements of the form (x,y) where x is a numpy array and y is -1 or 1
	each x is a feature vector of dimension 2
	each y is a label
theta_params is a tuple of the form (theta, theta_zero)
	theta is a 2-dimensional numpy array
	theta_zero is a number

return: None, but will plot the positive(+1 label) and negative(-1 label) training vectors and the boundary theta[0]*X + theta[1]*Y + theta_zero = 0 (if theta_params is not None, NOTE: X and Y here are the cartesian variables for the x and y axes)
'''
def plot_2D_training(Sn, theta_params=None):
	pos_x = [x[0] for (x,y) in Sn if y==1]	# get the x and y coordinates for the positive and negative training samples
	neg_x = [x[0] for (x,y) in Sn if y==-1]
	pos_y = [x[1] for (x,y) in Sn if y==1]
	neg_y = [x[1] for (x,y) in Sn if y==-1]
	plt.plot(pos_x, pos_y, "rs", ms=10)			# plot positive samples with a plus
	plt.plot(neg_x, neg_y, "bo", ms=10)			# plot negative samples with an "x"
	if theta_params is not None:			# if the theta params are avaiable, plot the decision boundary
		(theta, theta_zero) = theta_params
		X = [min(min(pos_x), min(neg_x))*1.5,max(max(pos_x), max(neg_x))*1.5]	# make sure the boundary covers the screen by finding the samples at the fringes of thes screen
		if theta[1] == 0:
			X = [theta_zero,theta_zero]
			Y = [10, -10]
		else:
			Y = [(-1.0*theta_zero - theta[0]*X[0])/theta[1], (-1.0*theta_zero - theta[0]*X[1])/theta[1]] # find their corresponding y coordinates
		plt.plot(X, Y, color="k", linestyle="-",linewidth=2)
	plt.grid()
	plt.show()


if __name__ == "__main__":
	A = np.array([[1,0,0],[0,1,0]])
	
	''' 
	Ignore this:
	Sn = [
	(np.array([1,2,2]),-1),
	(np.array([1,-2,2]),1)
	]

	Sn = [(A.dot(x),y) for (x,y) in Sn]
	'''

	Sn = [
	(np.array([1,2]),-1),
	(np.array([-1,-2]),1)
	]
	theta_params = perceptron(Sn, verbose=True, through_origin=True, max_iterations=100, start_with=1, plot_after_misclassification=False)
	plot_2D_training(Sn, theta_params)
	
