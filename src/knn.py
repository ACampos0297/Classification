import math

from data import Dataset, Labels
from utils import evaluate
import os, sys
import string

K = 1

class KNN:
	def __init__(self):
		# bag of words document vectors
		self.bow = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Save all the documents in the train dataset (ds) in self.bow.
		You need to transform the documents into vector space before saving
		in self.bow.
		"""
		for d in ds:
			docfrec = dict()
			string = d[1].lower()
			splits = string.split()
			for word in splits:
				if word in docfrec:
					docfrec[word] += 1.0
				else:
					docfrec[word] = 1.0
			self.bow.append((d[2], docfrec))

	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Predict class for x.
		1. Transform x to vector space.
		2. Find k nearest neighbors.
		3. Return the class which is most common in the neighbors.
		"""
		string = x.lower()
		splits = string.split()
		docfrec = dict()
		for word in splits:
			if word in docfrec:
				docfrec[word] += 1.0
			else:
				docfrec[word] = 1.0

		docfrecsq = dict()
		for word in docfrec:
			docfrecsq[word] = math.pow(docfrec[word], 2)

		#cosine similarities
		scores = dict()
		for doc in self.bow:
			similarity = sum(docfrec[key]*doc[1].get(key, 0) for key in docfrec)
			bsq = math.sqrt(sum(math.pow(doc[1].get(key, 0), 2) for key in docfrecsq))
			asq = math.sqrt(sum(docfrecsq.values()))
			similarity = similarity/float(float(asq)*bsq)
			scores[doc[0]] = similarity

		sortedscores = sorted(scores.items(), key=lambda x: x[1])
		return Labels(sortedscores[len(sortedscores)-1][0])

def main(train_split):
	knn = KNN()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	knn.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(knn, ds)

	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(knn, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(knn, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
