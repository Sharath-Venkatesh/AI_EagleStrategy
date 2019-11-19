#!/usr/bin/env python
# coding: utf-8

# In[26]:


from datetime import datetime
import random
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import time
import math

df = pd.read_csv('Eyes.csv')


# In[27]:


l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

X = df[eyes]
y = df['truth_value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import numpy as np
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# In[34]:



#Return a 1D numpy array containing randomly generated weights and biases.
def initialize():
    layer1=[]
    min_val=-5
    max_val=5
    for i in range(0,24):
        weight=[]
        for j in range(0,4):
            a=random.uniform(min_val,max_val)
            weight.append(a)
        layer1.append(weight)
    bias1=[]
    for i in range(0,4):
        c=random.uniform(min_val,max_val)
        bias1.append([c])
    layer2=[]
    for i in range(0,4):
        weight=[]
        for j in range(0,1):
            a=random.uniform(min_val,max_val)
            weight.append(a)
        layer2.append(weight)
    bias2=[]
    for i in range(0,1):
        c=random.uniform(min_val,max_val)
        bias2.append([c])
    initial_weights=[]

    initial_weights.append(np.array(layer1))
    initial_weights.append(np.array(bias1))
    initial_weights.append(np.array(layer2))
    initial_weights.append(np.array(bias2))
    
    l1=initial_weights[0]
    l1=l1.flatten().tolist()
    l2=initial_weights[1]
    l2=l2.flatten().tolist()
    l3=initial_weights[2]
    l3=l3.flatten().tolist()
    l4=initial_weights[3]
    l4=l4.flatten().tolist()
    initial_weights=l1+l2+l3+l4
    return initial_weights

#Randomly generate a neighbour.
def choose_neighbour(p):
    neigh=[0]*len(p)
    t=1
    for i in range(len(p)):
        neigh[i]=p[i]+random.uniform(-0.02,0.02)
    return neigh
#Return weights of the first layer
def first_layer(state):
    l1=state[0:96]
    b1=state[96:100]
    l1=np.reshape(l1,(-1,4))
    l1=[np.array(l1)]
    l1.append(np.array(b1))
    return l1
#Returns weights of the second layer
def second_layer(state):
    l2=state[100:104]
    b2=[state[104]]
    l2=np.reshape(l2,(-1,1))
    l2=[np.array(l2)]
    l2.append(np.array(b2))
    return l2
#Return the count of correctly classified instance and accuracy of model
def predict(model,x,y):
    y_train_pred = model.predict(x)
    # setting a confidence threshhold of 0.9
    y_pred_labels = list(y_train_pred > 0.9)
    #print(y_pred_labels)
    for i in range(len(y_pred_labels)):
        if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
        else : y_pred_labels[i] = 0
    count=0
    for i in range(len(y)):
        if(y[i]==y_pred_labels[i]):
            count=count+1
    acc1=count/len(y)
    return [count,acc1]


#SA
def simulated_annealing(model):
    acc1=0
    acc2=0
    neighbour=[]
    temp=1
    all_index=[]
    initial_weights=initialize()

    same_count=0
    flag=0
    for z in range(30):  
        #while(temp<=0.1):
        if(flag==0):
            initial_first_layer=first_layer(initial_weights)
            initial_second_layer=second_layer(initial_weights)

            model.layers[0].set_weights(initial_first_layer)
            model.layers[2].set_weights(initial_second_layer)

            predict_val1=predict(model,X_train,y_train)
            count1=predict_val1[0]
            acc1=predict_val1[1]

        flag=0

        neighbour=choose_neighbour(initial_weights)
        neigh_first_layer=first_layer(neighbour)
        neigh_second_layer=second_layer(neighbour)
        model.layers[0].set_weights(neigh_first_layer)
        model.layers[2].set_weights(neigh_second_layer)

        predict_val2=predict(model,X_train,y_train)

        count2=predict_val2[0]
        acc2=predict_val2[1]

        if(acc2-acc1<0.0001):
            same_count=same_count+1
        if(same_count==2):
            same_count=0
            old_weights=initial_weights
            while(1):
                new_rand_initial_weights=initialize()
                initial_first_layer=first_layer(new_rand_initial_weights)
                initial_second_layer=second_layer(new_rand_initial_weights)

                model.layers[0].set_weights(initial_first_layer)
                model.layers[2].set_weights(initial_second_layer)

                predict_val1=predict(model,X_train,y_train)
                count1=predict_val1[0]
                acc1=predict_val1[1]
                if(acc1>acc2):
                    all_index.append(acc1)
                    incr_acc.append(acc1)
                    initial_weights=new_rand_initial_weights
                    flag=1
                    break
            if(acc1>=0.93):#Cap for accuracy here
                break
            continue

        if(acc2>=acc1):
            initial_weights=neighbour
            all_index.append(acc2)
            incr_acc.append(acc2)
        else:

            delta=math.exp((acc2-acc1)/temp)
            x=random.uniform(0,1)

            if(x<=delta):

                initial_weights=neighbour
                incr_acc.append(acc2)
                all_index.append(acc2)
            else:
                flag=1
        temp=temp*(0.999)
    return initial_weights




from collections import Iterable
def my_flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in my_flatten(item):
                yield x
        else:        
            yield item

#Generate the best neighbour out of the five neighbours generated.
def hill_neighbour(t,p):
    no_neigh=5#Number of neighbours to be generated.
    neigh=[]
    for z in range(no_neigh):
        offset=random.uniform(-0.02,0.02)
        first_layer_weights =p[0][0]

        first_layer_weights=[ x+offset for x in first_layer_weights]
        first_layer_biases  = p[0][1]
        first_layer_biases=[x+offset for x in first_layer_biases]
        second_layer_weights = p[1][0]
        second_layer_weights=[x+offset for x in second_layer_weights]
        second_layer_biases  = p[1][1] 
        second_layer_biases=[x+offset for x in second_layer_biases]
        first=[first_layer_weights,first_layer_biases]
        second=[second_layer_weights,second_layer_biases]
        weights=[first,second]
        neigh.append(weights)
    acc=[]
    weight=[]
    for i in range(no_neigh):
        list1=list(my_flatten(neigh[i]))
        flw=first_layer(list1)
        slw=second_layer(list1)
        model.layers[0].set_weights(flw)
        model.layers[2].set_weights(slw)
        accuracy=predict(model,X_train,y_train)
        if(accuracy[1] not in acc):
            acc.append(accuracy[1])
            weight.append(neigh[i])

    max_neighbour=acc.index(max(acc))
    original=list(my_flatten(p))
    flw=first_layer(original)
    slw=second_layer(original)
    model.layers[0].set_weights(flw)
    model.layers[2].set_weights(slw)
    return weight[max_neighbour]
#Returns the weights of the model as a list.
def weights_given_model(model):
    first_layer_weights = model.layers[0].get_weights()[0]
    first_layer_biases  = model.layers[0].get_weights()[1]
    second_layer_weights = model.layers[2].get_weights()[0]
    second_layer_biases  = model.layers[2].get_weights()[1] 
    first=[first_layer_weights,first_layer_biases]
    second=[second_layer_weights,second_layer_biases]
    weights=[first,second]
    return weights


#Hill Climbing
def hill_climbing(model,global_accuracy):
    global_weights=weights_given_model(model)
    old_acc=global_accuracy
    prev_weights=weights_given_model(model)
    for iterations in range(100):
        prev_weights=weights_given_model(model)     
        
        best_neighbour=hill_neighbour(1,prev_weights)
        best_neighbour=list(my_flatten(best_neighbour))

        flw=first_layer(best_neighbour)
        slw=second_layer(best_neighbour)
        
        model.layers[0].set_weights(flw)
        model.layers[2].set_weights(slw)
        new_acc=predict(model,X_train,y_train)
        
        if(new_acc[1]>old_acc):
            incr_acc.append(new_acc[1])  
            old_acc=new_acc[1]           
            pass

        else:
            prev_weights=list(my_flatten(prev_weights))
            flw=first_layer(prev_weights)
            slw=second_layer(prev_weights)
            model.layers[0].set_weights(flw)
            model.layers[2].set_weights(slw)

            
    weights_accuracy=[]
    weights_accuracy.append(weights_given_model(model))
    pred_val=predict(model,X_train,y_train)
    weights_accuracy.append(pred_val[1])
    return weights_accuracy



model = Sequential()
model.add(Dense(4, input_dim = X_train.shape[1])) 
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
incr_acc=[]#For appending the accuracy for each epoch
start = time.time()
global_accuracy=-1
#Eagle Strategy starts here
while(global_accuracy<=0.96):
    incr_acc=[]
    weights=simulated_annealing(model)
    first_layer_w=first_layer(weights)
    second_layer_w=second_layer(weights)
    model.layers[0].set_weights(first_layer_w)
    model.layers[2].set_weights(second_layer_w)
    global_accuracy=predict(model,X_train,y_train)[1]
    pe=0.2
    if pe< random.uniform(0,1):
        weights_and_acc=hill_climbing(model,global_accuracy)
        hc_acc=weights_and_acc[1]
        if(hc_acc>global_accuracy):
            incr_acc.append(weights_and_acc[1])
            weights=weights_and_acc[0]
            global_accuracy=hc_acc

end = time.time()
print()
global_accuracy=predict(model,X_train,y_train)[1]
print("Train accuracy :",global_accuracy)
predict_val1=predict(model,X_test,y_test)
count1=predict_val1[0]
acc1=predict_val1[1]
print("Test Accuracy : ",acc1)
print("Train time",end - start)




# In[29]:


#Visualization of accuracy.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(incr_acc)
epoch=[x for x in range(len(incr_acc))]
plt.plot(epoch,incr_acc)


# In[ ]:





# In[ ]:




