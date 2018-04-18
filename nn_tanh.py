# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:34:31 2018

@author: Briti
"""

# program for implementation of ANN for classification of spam or ham

#numpy library for mathemetical operations
import numpy as np
#nltk library for preprocessing the data
import nltk
import random

'''Function to pre_process the data''' 
def pre_process(sentence):
    # list of words
    word_tokens = nltk.tokenize.word_tokenize(sentence)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.stem.porter.PorterStemmer()
    processed_sent = ""
    for token in word_tokens:
        if token not in unique_words:
            unique_words.add(token)
        if token not in stop_words:
            processed_sent+= stemmer.stem(token)
            processed_sent+= " "
    return processed_sent

'''Function to implement bag of words model'''
def vectorize_data(sentence,length_of_words):
    sentence_words = nltk.tokenize.word_tokenize(sentence)
    # frequency word count
    vector = np.zeros(length_of_words)
    for sw in sentence_words:
        for i,word in enumerate(unique_words):
            if word == sw:
                vector[i] = 1 
                break;
    return np.array(vector)
    
'''Tanh activation function'''
def tanh(x):
    return np.tanh(x)

'''Tanh derivative function'''
def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

        
def test(input_value,output_value,weight_hidden1,bias_hidden1,weight_hidden2,bias_hidden2,weight_output,bias_output): 
     #Forward Propogation
    hidden_layer_input1=np.dot(input_value,weight_hidden1)
    hidden_layer_input=hidden_layer_input1 + bias_hidden1
    hiddenlayer1_output = tanh(hidden_layer_input)
    hiddenlayer2_input = hiddenlayer1_output.dot(weight_hidden2)+bias_hidden2
    hiddenlayer2_output = tanh(hiddenlayer2_input)
    output_layer_input1=np.dot(hiddenlayer2_output,weight_output)
    output_layer_input= output_layer_input1+ bias_output
    output = tanh(output_layer_input)
    test_output=[]
    for x in output:
        if x>0.5:
            test_output.append(1)
        else:
            test_output.append(0)
    error = output_value-np.array(test_output).reshape(len(test_output),1)
    return np.sum(error)*-1 if np.sum(error)<0 else np.sum(error)
    
    
    

    
if __name__ == "__main__":
    
    #Variable declarations
    unique_words=set() #set of unique words
    sentence_to_encoding = [] #list to contain the encoding of each sentence
    
    #Initializing the network parameters
    lr=0.05 #Setting learning rate
    hiddenlayer1 = 100 #number of neurons in hidden layers one 
    hiddenlayer2 = 50 #number of neurons in hidden layers two
    output_neurons = 1 #number of neurons at output layer
    epoch=5000 #Setting training iterations
    
    np.random.seed(42)
        
    #List for maintaining input and output train and test data
    input_train = []
    input_test = []
    output_train = []
    output_test = []
    
    error_test = []
    error_train = []
    
    data = open("C:/Users/Briti/Assignment_2_data.txt","r")
    output = []
    sentence_list = []
    i=0
    for line in data:
        data = line.strip().split()
        if(data[0]=="ham"):
            output.append(1)
        else:
            output.append(0)
        #process the data
        processed_sentence = pre_process(' '.join(data[1:]))
        sentence_list.append(processed_sentence)
        
    #Rndomize the data for test train split    
    line = list(zip(output, sentence_list))
    random.shuffle(line)
    output, sentence_list = zip(*line)
        
    #80% test train split
    index_train = int(len(output)*0.80)
    index_test = int(len(output)*0.20)
    
    output_train = np.array(output[0:index_train]).reshape(index_train, 1)
    output_test = np.array(output[index_train+1:]).reshape(len(output[index_train+1:]), 1)
    
    #Represent the sentences as list of unique words
    length_of_words = len(unique_words)
    print(len(unique_words))
    for sent in sentence_list:
        sentence_to_encoding.append(vectorize_data(sent,length_of_words))
        
    input_train = np.array(sentence_to_encoding[0:index_train]).reshape(index_train, length_of_words)
    input_test = np.array(sentence_to_encoding[index_train+1:]).reshape(len(output[index_train+1:]), length_of_words)

    #Initializing the input layer
    inputlayer= input_train.shape[1]
    weight_hidden1 = 2*np.random.random((inputlayer,hiddenlayer1))-1
    #Random initialization of the weights
    bias_hidden1 = 2*np.random.random((1,hiddenlayer1))-1
    weight_hidden2 = 2*np.random.random((hiddenlayer1,hiddenlayer2))-1
    bias_hidden2 = 2*np.random.random((1,hiddenlayer2))-1
    weight_output = 2*np.random.random((hiddenlayer2,output_neurons))-1
    bias_output = 2*np.random.random((1,output_neurons))-1
    
    #Stochastic Gradient Descent
    for i in range(epoch):
        
        '''Forward Propagation'''
        
        j=random.randint(0,4458)
        
        X = (input_train[j].reshape((1,length_of_words)))
        hiddenlayer1_input = X.dot(weight_hidden1)+bias_hidden1
        hiddenlayer1_output = tanh(hiddenlayer1_input)
        
        hiddenlayer2_input = hiddenlayer1_output.dot(weight_hidden2)+bias_hidden2
        hiddenlayer2_output = tanh(hiddenlayer2_input)
        
        otputlayer_input = hiddenlayer2_output.dot(weight_output)+bias_output
        final_output = tanh(otputlayer_input)  
        
        #Error gradient
        error = output_train[j]-final_output
        
        
        #To test the error after every epoch
        error_tr = test(input_train,output_train,weight_hidden1,bias_hidden1,weight_hidden2,bias_hidden2,weight_output,bias_output)
        error_te = test(input_test,output_test,weight_hidden1,bias_hidden1,weight_hidden2,bias_hidden2,weight_output,bias_output) 
        error_test.append(error_te)
        error_train.append(error_tr)
        
        
        '''Backpropagation'''
        
        slope_output_layer = tanh_deriv(final_output)
        slope_hidden_layer1 = tanh_deriv(hiddenlayer1_output)
        slope_hidden_layer2 = tanh_deriv(hiddenlayer2_output)
        
        delta_output = error*slope_output_layer
        error_hidden_layer2 = delta_output.dot(weight_output.T)
        
        delta_hiddenlayer2 = error_hidden_layer2 * slope_hidden_layer2
        error_hidden_layer1 = delta_hiddenlayer2.dot(weight_hidden2.T)
        
        delta_hiddenlayer1 = error_hidden_layer1 * slope_hidden_layer1
        
        weight_output += hiddenlayer2_output.T.dot(delta_output) *lr
        bias_output += np.sum(delta_output, axis=0,keepdims=True) *lr
        
        weight_hidden2 += hiddenlayer1_output.T.dot(delta_hiddenlayer2) *lr
        bias_hidden2 += np.sum(delta_hiddenlayer2, axis=0,keepdims=True) *lr
        
        weight_hidden1 += X.T.dot(delta_hiddenlayer1) *lr
        bias_hidden1 += np.sum(delta_hiddenlayer1, axis=0,keepdims=True) *lr
        