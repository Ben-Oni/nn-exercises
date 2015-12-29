# trying to program 2-layer NN from memory as an exercise
# see https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def sigm(x):
	return 1/(1+np.exp(-x))

def sigmderiv(x):
	return x*(1-x)

# Training set
# Inputs   Output
# 0 0 1 -> 0
# 1 1 1 -> 0
# 1 0 1 -> 1
# 0 1 1 -> 1

inputs = np.array(
	[[0, 0, 1],
	 [0, 1, 1],
	 [1, 0, 1],
	 [1, 1, 1]])

targets = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

l1_weights = 2*np.random.random((3, 4))-1 # one row per incoming weight, one column vector per neuron
l2_weights = 2*np.random.random((4, 1))-1

for i in xrange(60000):
	# feed forward
	l1_rawoutputs = np.dot(inputs, l1_weights) # one column per neuron
	l1_outputs = sigm(l1_rawoutputs)

	l2_rawoutputs = np.dot(l1_outputs, l2_weights) # l1:(4x4)xl2:(4x1)=(4x1)
	l2_outputs = sigm(l2_rawoutputs)

	l2_errors = l2_outputs - targets # (4x1)
	
	if (i % 10000) == 0:
		print "Error: "+str(np.mean(np.abs(l2_errors)))
	
	# back propagation
	l2_deltaweights = l2_errors * sigmderiv(l2_outputs) # (4x1)
	l1_errors = np.dot(l2_deltaweights, l2_weights.T) #mistake: transposed l2-delta instead of weights, thinking it wouldn't matter which. (4x1).(1x4)=(1x1) != (1x4).(4x1)=(4x4)
	l1_deltaweights = l1_errors * sigmderiv(l1_outputs)
	l2_weights += -np.dot(l1_outputs.T, l2_deltaweights)
	l1_weights += -np.dot(inputs.T, l1_deltaweights)

print "Output after training:"
print l2_outputs

