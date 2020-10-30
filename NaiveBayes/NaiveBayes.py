import numpy
import random
import time
import os
from math import *

class NaiveBayes():
    def __init__(self, training_file, testing_file):
        self.training_file = training_file
        self.testing_file = testing_file

        self.train_set = numpy.loadtxt(self.training_file,dtype=float,delimiter=' ')
        self.test_set = numpy.loadtxt(self.testing_file,dtype=float,delimiter=' ')

        self.tp = 0 #True Positive
        self.fp = 0 #False Positive
        self.tn = 0 #True Negative
        self.fn = 0 #False Negative

        self.predictions = []
        self.actual = []

    def ClearMemory(self):
        self.predictions = []
        self.actual = []
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def ViewDict(self, d):
        for label in d:
            print(label)
            for row in d[label]:
                print(row)

        # calculate mean
    def CalcRowMean(self, row):
        return (sum(row)/float(len(row)))

        # calc standard deviation
    def CalcRowSD(self, row):
        avg = self.CalcRowMean(row)
        variance = sum([(x-avg)**2 for x in row]) / float(len(row)-1)
        return (variance**(1/2))

        #Calculate all the measures required..
    def SeparateByClass(self, dataset):
        separated = dict()
        for dr in dataset:
            # classifier is last item in row
            class_val = dr[-1]
            if (class_val not in separated):
                separated[class_val] = list()
            separated[class_val].append(dr)
        return separated

    def SummarizeDataset(self, dataset):
        sum = []
        # finally got to use the zip function :)
        for c in zip(*dataset):
            sum.append((self.CalcRowMean(c), self.CalcRowSD(c), len(c)))
        # delete classifier column
        del(sum[-1])
        return sum

    def SummarizeByClass(self, dataset):
        separated = self.SeparateByClass(dataset)
        summaries = dict()
        for class_val, rows in separated.items():
            summaries[class_val] = self.SummarizeDataset(rows)
        return summaries

    def CalculateProbability(self, x, mean, stdev):
        ex = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * ex

    def CalculateClassProbabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries ])
        probs = dict()
        for class_val, class_summaries in summaries.items():
            probs[class_val] = summaries[class_val][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, sd, count = class_summaries[i]
                probs[class_val] *= self.CalculateProbability(row[i], mean, sd)
        return probs

    def MakePrediction(self, summaries, row):
        probs = self.CalculateClassProbabilities(summaries, row)
        best_label, best_prob = None, -1.0
        for class_val, prob in probs.items():
            if best_label is None or prob > best_prob:
                best_prob = prob
                best_label = class_val
        return best_label

    def RunNaiveBayes(self):
        summarize = self.SummarizeByClass(self.train_set)
        # get actual classifications
        for row in self.test_set:
            out = self.MakePrediction(summarize, row)
            self.predictions.append(out)
            self.actual.append(row[-1])
        return self.predictions

    def Accuracy(self):
        top = self.tp + self.tn
        bottom = self.tp + self.fp + self.tn + self.fn
        return (top / bottom)

    def Sensitivity(self):
        bottom = self.tp + self.fn
        return (self.tp / bottom)

    def Specificity(self):
        bottom = self.fp + self.tn
        return (self.tn / bottom)

    def Precision(self):
        bottom = self.tp + self.fp
        return (self.tp / bottom)

    def GenerateTestResults(self, label):
        accuracy = str(self.Accuracy())
        sensitivity = str(self.Sensitivity())
        specificity = str(self.Specificity())
        precision = str(self.Precision())
        info =  "\nResults for {0}:\n\n".format(label)
        info += "Accuracy: {0}\n".format(accuracy)
        info += "Sensitivity/Recall: {0}\n".format(sensitivity)
        info += "Specificity: {0}\n".format(specificity)
        info += "Precision: {0}\n\n".format(precision)
        info += self.GetStats()
        return info

    def GetStats(self):
        info = "True Positives: {0}\n".format(self.tp)
        info += "False Positives: {0}\n".format(self.fp)
        info += "True Negatives: {0}\n".format(self.tn)
        info += "False Negatives: {0}\n\n".format(self.fn)
        return info

#  used to find tp,fp, ect
    def Test_NaiveBays(self):
        for i in range(len(self.predictions)):
            currPrediction = self.predictions[i]
            currActual = self.actual[i]
            if  currPrediction == 1 and currActual == 1:
                self.tp += 1
            elif currPrediction == 1 and currActual == -1:
                self.fp += 1
            elif currPrediction == -1 and currActual == -1:
                self.tn += 1
            elif currPrediction == -1 and currActual == 1:
                self.fn +=1
            else:
                print("[-] done fucked up")
