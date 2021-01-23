import numpy as np


class Evaluation():
	def __init__(self,n_classes):
		self.confusion_Matrix=np.zeros((n_classes,n_classes),dtype=int)

	def get_confusion_Matrix(self):
		return self.confusion_Matrix

	def add_prediction(self,pred,label):
		for j,i in zip(pred,label):
			self.confusion_Matrix[i,j] +=1

	def evaluate(self):
		
		True_Postive = self.confusion_Matrix.diagonal()
		True_Postive_And_False_Negative = self.confusion_Matrix.sum(axis=1)
		True_Postive_And_False_Postive = self.confusion_Matrix.sum(axis=0)

		Precision = True_Postive / True_Postive_And_False_Postive
		Recall = True_Postive / True_Postive_And_False_Negative

		F1 = 2*(Precision*Recall)/(Precision+Recall)

		return F1.mean()


