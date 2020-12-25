from abstract_classes import *

class Dense(Layer):

	def __init__(self,indim,outdim):
		super().__init__()
		self.init_weights(indim,outdim)

	def init_weights(self,dim):
		# xavier weight initialization
		self.weights['W'] = np.random.randn(indim,outdim) * np.sqrt( 2/(indim+outdim) )
		self.weights['b'] = np.zeros(1,outdim)

	def forward(self,X):

		output = np.dot(X , self.weights['W']) + self.weights['b']
		self.cache['X'] = X
		self.cache['output'] = output

		return output

	def backward(self,dY):

		dX = np.dot(dY,self.grad['X'].T)
		X  = self.cache['X']
		dW = np.dot(self.grad['W'],dY)
		db = np.sum(dY, axis = 0, keepdims = True)
		self.weights_global_grads = {'W': dW, 'b': db}
		return dX

	def calculate_local_grads(self,X):

		self.local_grads['X'] = self.weights['W']
		self.local_grads['W'] = X
		self.local_grads['b'] = np.ones_like(self.weight['b'])






