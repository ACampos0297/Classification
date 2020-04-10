from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""
		return Labels(0)


def main(train_split):
	nb = NaiveBayes()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	nb.train(ds)
	
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(nb, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(nb, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(nb, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
