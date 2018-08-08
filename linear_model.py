
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.datasets import make_multilabel_classification 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import preprocessing

import math
import tensorflow as tf
from tensorflow.python.data import Dataset




pd.options.display.max_rows = 20
pd.options.display.float_format = '{:.6f}'.format



#Загрузка данных

problem_train_features = pd.read_csv("D:/ds_problem/problem_train.csv", na_values="?", low_memory=False )
problem_train_labels = pd.read_csv("D:/ds_problem/problem_labels.csv", na_values="?", low_memory=False )
problem_test_features = pd.read_csv("D:/ds_problem/problem_test.csv", na_values="?", low_memory=False )


#Объединение тренировочной и тестовой выборки для обработки данных
problem_full_features = pd.concat([problem_train_features,problem_test_features],axis=0)
problem_full_features = problem_full_features.drop_duplicates(subset='id')


#Функция предварительной обработки данных. Сокращает размерность пространства признаков до 25 путем отбрасывания
# признаков с неопределнными значениями (т.к нет информации, что конкретно означают признаки и какие значения
# можно было бы подставить вместо NaN, было принято такое решение) и признаков, имеющих константное значение на всей выборке.
def process_features(features):
    
    #удаляем все series имеющие хотя бы 1 NaN значение признака
    features = features.dropna(axis=1, how='any')
    
    #разбиваем dataframe на два, в одном - столбцы типа object, в другом - числовые значения 
    subframe_type_object = features.select_dtypes(include=['object']).copy()
    subframe_type_num = features.select_dtypes(exclude=['object']).copy()
    
    #для всех всех столбцов с буквами применяем one hot encoding
    subframe_type_object_encoded = pd.DataFrame()
    for serie in subframe_type_object.columns:
            one_hot_columns = pd.get_dummies(subframe_type_object[serie], prefix=serie)
            subframe_type_object_encoded=pd.concat([subframe_type_object_encoded, one_hot_columns], axis=1)
   
    #из dataframe с числами удаляем столбцы с постоянным значением (дисперсия==0.) и столбец id  
    subframe_type_num = subframe_type_num.drop(axis=1,columns=['id'])
    subframe_type_num = subframe_type_num.loc[:, subframe_type_num.var() != 0.0]
   
    #масштабируем численные значения на отрезок [0,1] методом min-max scaling
    subframe_type_num_scaled = pd.DataFrame()
    subframe_type_num_scaled=(subframe_type_num-subframe_type_num.min())/(subframe_type_num.max()-subframe_type_num.min())
    
    #объединяем датафреймы
    processed_features = pd.concat([subframe_type_num_scaled,subframe_type_object_encoded], axis=1)
    
    #снова разбиваем на train и test выборки
    processed_train_features = processed_features.head(8000)
    processed_test_features = processed_features.tail(2000)
    return  processed_train_features, processed_test_features


#Предварительная обработка датафреймов с признаками и метками
training_features, test_features = process_features(problem_full_features)
#Столбец id не нужен
training_labels = problem_train_labels.drop(axis=1,columns=['id'])


#извлечение данных из dataframe в numpy array
x_train = training_features.values
y_train = training_labels.values
x_test = test_features.values


# Определение модели - логистическая регрессия
def model(X, W, B): 

    return tf.nn.sigmoid(tf.matmul(X, W)+ B) 


#Инициализация весов - случаные значения из нормального распределения
def init_weights(shape): 
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) 


#learning rate, кол-во шагов обучения (training_epochs),
learning_rate = 0.05 
training_epochs = 2500
batch_size = 800

#кол-во входов и выходов(классов),суммарное количество батчей
num_input = x_train.shape[1] 
num_classes = y_train.shape[1] 
num_batches = int(x_train.shape[0]/batch_size) 

#Объявление модели
x = tf.placeholder("float", [None, num_input]) 
y = tf.placeholder("float", [None, num_classes]) 
b = tf.Variable(tf.zeros([num_classes]))
w = init_weights([num_input, num_classes]) 
predictions = model(x, w, b) 

#Функция ошибки - log loss. Reduction.NONE - рассчитывается отдельно для каждого класса
loss = tf.losses.log_loss(labels=y, predictions=predictions, epsilon=0.00000001, reduction=tf.losses.Reduction.NONE)

#Для оптимизации считается сумма значений по каждому классу для текущего батча,
# а затем берется среднее арифметическое по классам
loss_sum = tf.reduce_mean(tf.reduce_sum(loss,axis=0))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss_sum) 

init = tf.global_variables_initializer() 

with tf.Session() as sess:
    sess.run(init) 
    sess.run(tf.local_variables_initializer()) 
    
    for epoch in range(training_epochs): 
        #случайная выборка тренировочных примеров в текущий батч
        indices = np.random.choice(num_input, batch_size)
        x_batch, y_batch = x_train[indices], y_train[indices]   
        _, log_loss, l_sum = sess.run([optimizer, loss, loss_sum], 
                              feed_dict = {x : x_batch, y : y_batch})
        
        print("Log loss for %d epoch %.3f" %(epoch,l_sum)) #Вывод текущей ошибки
    print("\n")
    res_loss = np.sum(log_loss,axis=0)
    for cl, l_loss in enumerate(res_loss): #вывод ошибок по каждому классу по итогам обучения на последнем батче
        print("Log loss for %d class = %.3f" %(cl,l_loss))
     
    print("FINAL MEAN LOSS = %.3f (batch size = %d)" %(l_sum,batch_size)) #итоговая log_loss (средняя по всем классам по итогам обучения на последнем батче)

    sess.run(tf.local_variables_initializer())

    #Вероятности принадлежности классам для примеров из тестовой выборки
    probabilities = sess.run(predictions, feed_dict = {x : x_test})


#Формирование Dataframe и запись в файл
results = pd.DataFrame()
results['id'] = problem_test_features['id']
probs_df=pd.DataFrame(probabilities, columns=['class_%d' %index for index in range(14)])
results = pd.concat([results, probs_df] , axis=1)
results.to_csv('D:/ds_problem/problem_test_labels.csv', sep=',')


