# tried to program 1-neuron NN from memory as an exercise
# see https://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np #mistake: used require instead of import

def sigm(x): #mistake: used sigm = function(x) instead of def sigm(x)
	return 1/(1+np.exp(-x)) # mistake: used x instead of -x

def sigmdev(x):
	return x*(1-x) #correct


# Training set
# Inputs   Output
# 0 0 1 -> 0
# 1 1 1 -> 1
# 1 0 1 -> 1
# 0 1 1 -> 0

inputs = np.array(
	[[0, 0, 1],
	 [1, 1, 1],
	 [1, 0, 1],
	 [0, 1, 1]]) #correct

targets = np.array([[0, 1, 1, 0]]).T #mistake: only made 1D-array (one level of square brackets) instead of 2D (two pairs of square brackets).

np.random.seed(1) #minor mistake: forgot to seed

l1_weights = 2*np.random.random((3, 1))-1 #mistake: created 1x3 matrix instead of 3x1; minor mistake: didn't center around 0

for i in xrange(1000): #mistake: forgot about python's ranges in for loops
	# forward propagation
	l1_rawoutputs = np.dot(inputs, l1_weights) #correct
	l1_outputs = sigm(l1_rawoutputs) #correct
	l1_errors = l1_outputs - targets #correct; column vector

	# back propagation
	l1_deltaweights = l1_errors * sigmdev(l1_outputs) #mistake: calculated sigmdev(err) instead of multiplying error with slope

	l1_weights += -np.dot(inputs.T, l1_deltaweights) #mistake: inputs is not a column vector! But result has to be (4x1). (3x4).(4x1)=(4x1)

print "Output after training:"
print l1_outputs

