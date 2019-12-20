
"""
This program implements a fully connected neural network.
It uses different activation functions (softmax, tanh) for different layers.
This also implements gradient descent as a means to adjust the weights 
of the connections.
"""

import numpy as np

def mlp_cost(x, y, ws, bs, phis, alpha):
	epsilon = 1*10**-8 #prevents underflow -- log(0)
	numColumns = len(x)
	loop = len(ws)
	p = mlp_predict_proba(x, ws, bs, phis)


	firstSum = (-1/numColumns) * np.sum(y * np.log(p + epsilon))
	secondSum = 0
	for i in range(0,loop):
		secondSum += np.linalg.norm(ws[i])**2
	secondSum *= (alpha / 2) 
	return firstSum + secondSum


def mlp_propagate_error(x,y,ws,bs,phis,hs):

	numColumns = len(x)
	p = mlp_predict_proba(x, ws, bs, phis)
	lastGrad = (1/numColumns) * (p - y)
	ds = [lastGrad]
	ws = np.array(ws)
	
	for i in range(len(hs) - 2, 0, -1 ):
		d = (ds[0] @ ws[i].T) * (1 - hs[i]**2)
		ds.insert(0,d)
	
	return ds



def mlp_gradient(x, y, ws, bs, phis, alpha):
	m = len(x)
	hs = mlp_feed_forward(x, ws, bs, phis)
	ds = mlp_propagate_error(x,y,ws,bs,phis,hs)

	gradwJs = []
	gradbJs = []
	for i in range(0,len(hs) - 1, 1):
		# print(np.array(ds[i+1]).shape)
		# print(np.array(hs[i]).T.shape)
		# print(np.array(ws[i+1]).shape)
		gradwJ = (np.array(hs[i]).T @ np.array(ds[i])) + (alpha * np.array(ws[i]))
		
		gradwJs.append(gradwJ)
		gradbJ = np.ones(m).T @ ds[i]
		gradbJs.append(gradbJ)

	return gradwJs, gradbJs

def mlp_initialize(layer_widths):
	n = len(layer_widths)
	
	phi = []
	ws = []
	bs = []
	for i in range(n):
		if i < n - 2:
			phiF = mlp_tanh
		else:
			phiF = mlp_softmax
		phi.append(phiF)
	for i in range(n - 1):
		ws.append(np.random.normal(0,0.1,(layer_widths[i],layer_widths[i+1])))
		bs.append(np.random.normal(0,0.1,(1,layer_widths[i+1])))

	return ws,bs,phi

def mlp_gradient_descent(x, y, ws0, bs0, phis, alpha, eta, n_iter):
	w0 = []
	b0 = []
	for w in ws0:
		w0.append(w.copy())
	for b in bs0:
		b0.append(b.copy())
	for i in range(n_iter):
		#print(mlp_cost(x, y, w0, b0, phis, alpha))
		w, b = mlp_gradient(x, y, w0, b0, phis, alpha)
		for j in range(len(w)):
			w0[j] -= eta*w[j]
			b0[j] -= eta*b[j]
	return w0, b0


def mlp_run_mnist(eta=0.2, alpha=0.05, hidden_widths=[450], n_iter=300):
	x_training, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_data_prep()
	ws0, bs0, phis = mlp_initialize([len(x_training[0]),hidden_widths[0],10])

	ws_hat, bs_hat = mlp_gradient_descent(x_training, y_matrix_train, ws0, bs0, phis, alpha, eta, n_iter)
	
	test_acc = np.where(mlp_predict(x_test, ws_hat, bs_hat, phis) == y_test, 1, 0)
	train_acc = np.where(mlp_predict(x_training, ws_hat, bs_hat, phis) == y_train, 1 , 0)
	test_acc = np.mean(test_acc)
	train_acc = np.mean(train_acc)

	return x_training, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_acc, test_acc


##previous assignment

def mlp_check_dimensions(x,y,w,bs):
	m = len(x)
	n_prev = len(x[0])

	if len(w) != len(bs):
		return False

	for i in range(len(w)):
		n = len(w[i][0])

		if (n_prev != len(w[i])):
			return False
		if (n != len(bs[i][0])):
			return False
		if len(bs[i]) != 1:
			return False
		n_prev = n

	if (m != len(y) or n != len(y[0])):
		return False
	return True

def mlp_net_input(h,w,b):
	return h @ w + b

def mlp_tanh(z):
	return np.tanh(z)

def mlp_softmax(z):
	z = np.array(z)
	z = z.T
	func = z - np.max(z)
	func = np.exp(func)
	suma = np.sum(func,axis=0)
	
	final = func / suma
	return final.T

def mlp_feed_layer(h, w, b, phi):
	return phi(mlp_net_input(h,w,b))

def mlp_feed_forward(x, ws, bs,  phis):
	h = np.array(x)
	ws = np.array(ws)
	hs = [h]
	for i, w in enumerate(ws):
		h = mlp_feed_layer(h, w, bs[i], phis[i])
		hs.append(h)
	return hs

def mlp_predict_proba(x, ws, bs, phis):
	return (mlp_feed_forward(x, ws, bs,  phis)[-1])	

def mlp_predict(x, ws, bs, phis):
	return np.argmax(mlp_predict_proba(x, ws, bs, phis),axis=1)

def mlp_data_prep():
	x = np.load('x_mnist1000.npy')
	y = np.load('y_mnist1000.npy')
	m = len(x)
	c = 10

	y_matrix = np.zeros((m, c))
	y_matrix[(range(m), y.astype('int'))] = 1

	assert len(x) == len(y)
	np.random.seed(1)
	p = np.random.permutation(len(x))
	x = x[p]
	y = y[p]

	y_matrix = np.zeros((m, c))
	y_matrix[(range(m), y.astype('int'))] = 1
	
	nr_training = int(len(x) * 0.8)
	x_training = x[:nr_training, :]
	y_training = y[:nr_training]
	x_test = x[nr_training:, :]
	y_test = y[nr_training:]
	
	y_matrix_training = y_matrix[:nr_training]
	y_matrix_test = y_matrix[nr_training:]

	mean_training = x_training.mean(axis=0)
	std_dev_training = x_training.std(axis=0)
	std_dev_training = np.where(std_dev_training != 0, std_dev_training, 1)

	x_training = (x_training - mean_training) / std_dev_training
	x_test = (x_test - mean_training) / std_dev_training

	return x_training, x_test, y_training, y_test, y_matrix_training, y_matrix_test


