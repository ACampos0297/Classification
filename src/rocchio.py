from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class Rocchio:
    def __init__(self):
        # centroids vectors for each Label in the training set.
        self.centroids = dict()
        # document totals for each label
        self.totaldocs = dict()

    def train(self, ds):
        """
        ds: list of (id, x, y) where id corresponds to document file name,
        x is a string for the email document and y is the label.

        Loop over all the samples in the training set, convert the
        documents to vectors and find the centroid for each Label.
        """
        # convert to raw term frequencies
        for d in ds:
            docfrec = dict()
            if d[2] in self.totaldocs:
                self.totaldocs[d[2]] += 1.0
            else:
                self.totaldocs[d[2]] = 1.0
            string = d[1].lower()
            splits = string.split()
            for word in splits:
                if word in docfrec:
                    docfrec[word] += 1.0
                else:
                    docfrec[word] = 1.0
            # normalize the vector
            norm = math.sqrt(sum(math.pow(docfrec[key], 2) for key in docfrec))
            for word in docfrec:
                docfrec[word] = docfrec[word]/norm

            # add vector to centroid sum
            if d[2] not in self.centroids:
                self.centroids[d[2]] = docfrec
            else:
                for word in docfrec:
                    if word in self.centroids[d[2]]:
                        self.centroids[d[2]][word] += docfrec[word]
                    else:
                        self.centroids[d[2]][word] = docfrec[word]

        # calculate centroid by dividing by total num docs in label
        for label in Labels:
            for word in self.centroids[label]:
                self.centroids[label][word] = self.centroids[label][word]/self.totaldocs[label]

    def predict(self, x):
        """
        x: string of words in the document.

        Convert x to vector, find the closest centroid and return the
        label corresponding to the closest centroid.
        """
        docfrec = dict()
        string = x.lower()
        splits = string.split()
        for word in splits:
            if word in docfrec:
                docfrec[word] += 1.0
            else:
                docfrec[word] = 1.0
        # normalize the vector
        norm = math.sqrt(sum(math.pow(docfrec[key], 2) for key in docfrec))
        for word in docfrec:
            docfrec[word] = docfrec[word] / norm

        #calculate cosine similarities for each label
        docfrecsq = dict()
        for word in docfrec:
            docfrecsq[word] = math.pow(docfrec[word], 2)

        # cosine similarities
        scores = dict()
        for label in Labels:
            similarity = sum(docfrec[key] * self.centroids[label].get(key, 0)
                             for key in docfrec)
            asq = math.sqrt(sum(docfrecsq.values()))
            bsq = math.sqrt(sum(math.pow(
                                self.centroids[label].get(key, 0), 2) for key in docfrecsq))
            similarity = similarity / float(float(asq) * bsq)
            scores[label] = similarity
        sortedscores = sorted(scores.items(), key=lambda x: x[1])
        return Labels(sortedscores[len(sortedscores) - 1][0])

def main(train_split):
    rocchio = Rocchio()
    ds = Dataset(train_split).fetch()
    val_ds = Dataset('val').fetch()
    rocchio.train(ds)

    # Evaluate the trained model on training data set.
    print('-'*20 + ' TRAIN ' + '-'*20)
    evaluate(rocchio, ds)
    # Evaluate the trained model on validation data set.
    print('-'*20 + ' VAL ' + '-'*20)
    evaluate(rocchio, val_ds)

    # students should ignore this part.
    # test dataset is not public.
    # only used by the grader.
    if 'GRADING' in os.environ:
        print('\n' + '-'*20 + ' TEST ' + '-'*20)
        test_ds = Dataset('test').fetch()
        evaluate(rocchio, test_ds)

if __name__ == "__main__":
    train_split = 'train'
    if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
        train_split = 'train_half'
    main(train_split)
