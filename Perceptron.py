# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:52:51 2021

@author: Arthur
"""


from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt

def examples(fichier):
    f = open(fichier, 'r', encoding='utf-8')
    data = f.read()
    words_labels = []
    for bloc in data.split('\n\n'):
        words_list = []
        labels_list = []
        for line in bloc.split('\n'):
            if not line.startswith('#') and '-' not in line.split('\t')[0] and line!='':
                    words_list.append(line.split('\t')[2])
                    labels_list.append(line.split('\t')[3])
        words_labels.append((words_list, labels_list))
    return words_labels



train = examples("C:/Users/33623/Documents/Cours/NLP/Perceptron/fr_gsd-ud-train.conllu")
test = examples("C:/Users/33623/Documents/Cours/NLP/Perceptron/fr_gsd-ud-test.conllu")

def distrib(examples):
    dico = defaultdict(int)
    for words,labels in examples :
        for label in labels :
            dico[label]+=1
    return dico

def plotdistrib(distrib):
    pairs = sorted(distrib.items())
    x,y = zip(*pairs)
    plt.bar(x, y)
    plt.xlabel("label")
    plt.ylabel("occurences")
    plt.show()


def count_examples(examples):
    counter = 0
    for example in examples :
        counter+=1
    return counter

def is_low_up(word):
    if word[0].isupper():
        return "starts_with_upper" 
    else : 
        return "starts_with_lower"

def contains_digit(word):
    for char in word :
        if char.isdigit():
            return "contains_digit"
    return "not_contains_digit"


def features(examples):
    features_label = []
    for words,labels in examples:
        words.insert(0, "d")
        words.insert(1, "d")
        words.insert(len(words)+1, "f")
        words.insert(len(words)+2, "f")
        labels.insert(0, "d")
        labels.insert(1, "d")
        labels.insert(len(labels)+1, "f")
        labels.insert(len(labels)+2, "f")
        for i,w in enumerate(words):
            features = []
            if w!="d" and w!="f":
                features+=["curr_word_" + w, "prev_word_" + words[i-2],
                "prev_prev_words_" + words[i-3], "next_word_" + words[i],
                "next_next_word_" + words[i+1], "biais", is_low_up(w), contains_digit(w)]
            if features !=[] :
                features_label.append((features, labels[i]))
    return features_label

    

class vanillaPerceptron:
    def __init__(self):
        self.train_features_label_tuples = features(train) 
        self.test_features_label_tuples = features(test)
        self.parameters = defaultdict(lambda: defaultdict(int))
            
        
    def predict(self,obs_features):
        best_label = "NOUN"
        maxval = 0
        for label in self.parameters.keys():
            xi = [1 for feat in obs_features]
            wyi = [self.parameters[label][feat] for feat in obs_features]
            dot = np.dot(xi,wyi)
            if dot > maxval:
                maxval = dot
                best_label = label
        return best_label
        
    def train(self,n_epoch):
        for i in range(n_epoch):
            random.shuffle(self.train_features_label_tuples)
            for features,gold in self.train_features_label_tuples:
                pred = self.predict(features)
                if pred != gold:
                    for feat in features:
                        self.parameters[gold][feat] += 1
                        self.parameters[pred][feat] -= 1
    
    def evaluation(self):
        score = 0
        total = 0
        
        for i in range(len(self.test_features_label_tuples)):
            features,gold = self.test_features_label_tuples[i]
            pred = self.predict(features)
            if pred == gold:
                score += 1
            total +=1
        return (score/total)
    
class averagedPerceptron(vanillaPerceptron):
    def __init__(self):
        self.train_features_label_tuples = features(train) 
        self.test_features_label_tuples = features(test)
        self.parameters = defaultdict(lambda: defaultdict(int))
        self.a = defaultdict(lambda: defaultdict(int))
        self.last_update = defaultdict(int)
        
    def train(self,n_epoch):
        for i in range(n_epoch):
            random.shuffle(self.train_features_label_tuples)
            n_examples = 0
            for features,gold in self.train_features_label_tuples:
                n_examples += 1
                pred = self.predict(features)
                if pred != gold:
                    for feat in features:
                        self.a[gold][feat] += (n_examples - self.last_update[gold, feat]) * self.parameters[gold][feat]
                        self.last_update[gold, feat] = n_examples
                        self.a[pred][feat] += (n_examples - self.last_update[pred, feat]) * self.parameters[pred][feat]
                        self.last_update[pred, feat] = n_examples

                      
                        self.parameters[gold][feat] += 1
                        self.parameters[pred][feat] -= 1

        for label in self.a:
            for feat in self.a[label]:
                self.a[label][feat] += (n_examples - self.last_update[label, feat]) * self.parameters[label][feat]

    
    def post_train_predict(self,obs_features):
        return self.predict(obs_features)
        best_label = "NOUN"
        maxval = 0
        for label in self.a.keys():
            xi = [1 for feat in obs_features]
            wyi = [self.a[label][feat] for feat in obs_features]
            dot = np.dot(xi,wyi)
            if dot > maxval:
                maxval = dot
                best_label = label
        return best_label
    
    def evaluation(self):
        score = 0
        total = 0
        
        for i in range(len(self.test_features_label_tuples)):
            features,gold = self.test_features_label_tuples[i]
            pred = self.post_train_predict(features)
            if pred == gold:
                score += 1
            total +=1
        return (score/total)


for n in [1,2,5,10]:
    v = vanillaPerceptron()
    v.train(n)
    print("vanilla : " + str(n) + " epochs --> " + str(v.evaluation()))   

for n in [1,2,5,10]:
    a = averagedPerceptron()
    a.train(n)
    print("moyenné : " + str(n) + " epochs --> " + str(a.evaluation()))   






