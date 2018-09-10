
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy import sparse

pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 20


# Загружаем датасет, делим на тренировочную и тестовую выборки

# In[2]:


data = pd.read_csv("D:/renessans.txt", delimiter=';', index_col='POLICY_ID')
train_df = data[data['DATA_TYPE']=='TRAIN'].drop('DATA_TYPE', axis=1)
test_df = data[data['DATA_TYPE']=='TEST '].drop('DATA_TYPE', axis=1)
x_train = train_df.drop('POLICY_IS_RENEWED', axis=1)
y_train = train_df['POLICY_IS_RENEWED']
x_test = test_df.drop('POLICY_IS_RENEWED', axis=1)


# Посмотрим на корелляционную матрицу признаков

# In[3]:


x_train.corr()


# Ввиду сильной корелляции между признаками POLICY_BEGIN_MONTH и POLICY_END_MONTH, можно сделать вывод, что полисы в подавляющем большинстве выдаются на целое число лет.
#  В то же время, сезон выдачи полиса вряд ли как-то влияет на принятие клиентом решения о продлении полиса, так что данные признаки рассматривать не будем.

# In[4]:


x_train = x_train.drop(['POLICY_END_MONTH', 'POLICY_BEGIN_MONTH'], axis=1)


# In[5]:


print(x_train['POLICY_SALES_CHANNEL'].unique())
print(x_train['POLICY_SALES_CHANNEL_GROUP'].unique())


# POLICY_SALES_CHANNEL, POLICY_SALES_CHANNEL_GROUP - категориальные признаки. POLICY_SALES_CHANNEL принимает достаточно много различных значений, при этом POLICY_SALES_CHANNEL_GROUP - это сгруппированные значения данного признака.
#  В связи с этим POLICY_SALES_CHANNEL рассматривать нне будем, а  POLICY_SALES_CHANNEL_GROUP необходимо будет разбить на несколько признаков по категориям, используя one hot encoding.

# In[6]:


x_train = x_train.drop(['POLICY_SALES_CHANNEL'], axis=1)


# In[7]:


print(x_train['POLICY_BRANCH'].unique())


# POLICY_BRANCH - категориальный признак, принимающий 2 значения. Закодируем значения признака числами

# In[8]:


x_train['POLICY_BRANCH'] = [0 if (x == 'Москва') else 1 for x in x_train['POLICY_BRANCH'].values]


# In[9]:


print(x_train['POLICY_MIN_AGE'].unique())


# POLICY_MIN_AGE - числовой признак. Конкретный возраст вряд ли сильно кореллирует с вероятностью пролонгации полиса.
#  Однако если разбить людей по возрастным группам (например, "18-25 лет", "25-40 лет", "40-65 лет", "65+ лет"), можно будет проследить зависимости.

# In[10]:


x_train['AGE_18-25'] = [1 if (x<25) else 0 for x in x_train['POLICY_MIN_AGE'].values]
x_train['AGE_25-40'] = [1 if (x>=25)and(x<=40) else 0 for x in x_train['POLICY_MIN_AGE'].values]
x_train['AGE_40-65'] = [1 if (x>=40)and(x<65) else 0 for x in x_train['POLICY_MIN_AGE'].values]
x_train['AGE_65+'] = [1 if (x>=65) else 0 for x in x_train['POLICY_MIN_AGE'].values]
x_train = x_train.drop(['POLICY_MIN_AGE'], axis=1)


# In[11]:


print(x_train['POLICY_MIN_DRIVING_EXPERIENCE'].unique())


# POLICY_MIN_DRIVING_EXPERIENCE - числовой признак. Вероятно, в некоторых местах указан год начала вождения вместо опыта.
#  Исправим (будем считать, что данные за 2016 год, судя по времени создания файла с описанием задания). Кроме того, лучше выделить несколько групп водителей с разным стажем, аналогично тому, как это сделано для возраста.

# In[12]:


x_train['POLICY_MIN_DRIVING_EXPERIENCE'] = [(2016-x) if (x > 100) else x for x in x_train['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_train['DRV_EXP_0-3'] = [1 if (x<3) else 0 for x in x_train['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_train['DRV_EXP_3-10'] = [1 if (x>=3) & (x<10) else 0 for x in x_train['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_train['DRV_EXP_10-20'] = [1 if (x>=10) & (x<20) else 0 for x in x_train['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_train['DRV_EXP_20+'] = [1 if (x>=20) else 0 for x in x_train['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_train = x_train.drop(['POLICY_MIN_DRIVING_EXPERIENCE'], axis=1)


# VEHICLE_MAKE и VEHICLE_MODEL - категориальные признаки. Их кодирование с помощью one hot encoding даст нам большое количество признаков со множеством нулей.
#  Однако решение о пролонгации полиса зависит от марки и модели машины, поэтому попробуем их оставить и применим к ним one hot encoding (результаты тестирования над данными с учетом данных признаков дают прирост оценки accuracy на 0.05, так что оставим их при обучении итоговой модели)

# In[13]:


#x_train = x_train.drop(['VEHICLE_MAKE', 'VEHICLE_MODEL'], axis=1)


# VEHICLE_IN_CREDIT - категориальный, значения 0 и 1, оставляем как есть.

# VEHICLE_ENGINE_POWER, VEHICLE_SUM_INSURED - численные, необходимо отмасштабировать

# POLICY_INTERMEDIARY - категориальный признак с 1333 различных значений. Обойдемся без него.

# In[15]:


x_train = x_train.drop(['POLICY_INTERMEDIARY'], axis=1)


# INSURER_GENDER - категориальный. 2 значения. Сделаем из него бинарный признак со значениями 0 и 1

# In[16]:


x_train['INSURER_GENDER'] = [0 if (x == 'M') else 1 for x in x_train['INSURER_GENDER'].values]


# In[17]:


print(x_train['POLICY_CLM_N'].unique())
print(x_train['POLICY_CLM_GLT_N'].unique())
print(x_train['POLICY_PRV_CLM_N'].unique())
print(x_train['POLICY_PRV_CLM_GLT_N'].unique())


# POLICY_CLM_N, POLICY_CLM_GLT_N, POLICY_PRV_CLM_N, POLICY_PRV_CLM_GLT_N - категориальные, необходим one hot encoding
# В данных признаках неопределенные значения обозначаются по-разному (как N и n/a). Необходимо привести к общему значению.

# In[18]:


x_train['POLICY_CLM_N'] = ['N' if (x=='n/d') else x for x in x_train['POLICY_CLM_N'].values]
x_train['POLICY_CLM_GLT_N'] = ['N' if (x=='n/d') else x for x in x_train['POLICY_CLM_GLT_N'].values]


# CLIENT_HAS_DAGO, CLIENT_HAS_OSAGO, POLICY_COURT_SIGN, POLICY_HAS_COMPLAINTS - бинарные признаки, обработка не требуется

# CLAIM_AVG_ACC_ST_PRD - числовой, необходимо масштабирование

# POLICY_YEARS_RENEWED_N - числовой. В 46 примерах не определено значение. Установим для данных примеров значение 1, как наиболее популярный вариант.

# In[19]:


x_train['POLICY_YEARS_RENEWED_N'] = [1 if x=='N' else int(x) for x in x_train['POLICY_YEARS_RENEWED_N'].values]


# POLICY_DEDUCT_VALUE - числовой, потребуется масштабирование

# CLIENT_REGISTRATION_REGION - категориальный, 83 значения. One hot encoding

# POLICY_PRICE_CHANGE - числовой, потребуется масштабирование

# Применим операции масштабирования и кодирования признаков для тренировочного набора данных:

# In[20]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer


# In[21]:


min_max_scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder()
label_binarizer = LabelBinarizer()
label_binarizer_region = LabelBinarizer()
label_binarizer_make = LabelBinarizer()
label_binarizer_model = LabelBinarizer()


# In[22]:


numeric_features = ['VEHICLE_ENGINE_POWER', 'VEHICLE_SUM_INSURED', 'CLAIM_AVG_ACC_ST_PRD',
                    'POLICY_YEARS_RENEWED_N',  'POLICY_DEDUCT_VALUE', 'POLICY_PRICE_CHANGE']

binary_features = ['POLICY_BRANCH', 'INSURER_GENDER', 'AGE_18-25', 'AGE_25-40', 'AGE_40-65', 'AGE_65+',
                   'DRV_EXP_0-3', 'DRV_EXP_3-10', 'DRV_EXP_10-20', 'DRV_EXP_20+',
                   'VEHICLE_IN_CREDIT', 'CLIENT_HAS_DAGO', 'CLIENT_HAS_OSAGO',
                   'POLICY_COURT_SIGN', 'POLICY_HAS_COMPLAINTS']

category_num_features = ['POLICY_SALES_CHANNEL_GROUP'] 


# In[23]:


x_numeric = min_max_scaler.fit_transform(x_train[numeric_features])
x_binary = x_train[binary_features]
x_num_category = one_hot_encoder.fit_transform(x_train[category_num_features])
x_cat_CLM_N = label_binarizer.fit_transform(x_train['POLICY_CLM_N'])
x_cat_CLM_GLT_N = label_binarizer.transform(x_train['POLICY_CLM_GLT_N'])
x_cat_PRV_CLM_N = label_binarizer.transform(x_train['POLICY_PRV_CLM_N'])
x_cat_PRV_CLM_GLT_N = label_binarizer.transform(x_train['POLICY_PRV_CLM_GLT_N'])
x_cat_CR_REG = label_binarizer_region.fit_transform(x_train['CLIENT_REGISTRATION_REGION'])
x_cat_make = label_binarizer_make.fit_transform(x_train['VEHICLE_MAKE'])
x_cat_model = label_binarizer_model.fit_transform(x_train['VEHICLE_MODEL'])


# In[24]:


x_train_final = sparse.hstack((x_numeric, x_binary, x_num_category, x_cat_CLM_N, 
                               x_cat_CLM_GLT_N, x_cat_PRV_CLM_N, x_cat_PRV_CLM_GLT_N,
                               x_cat_CR_REG, x_cat_make, x_cat_model))


# Создадим валидационную выборку для отладки модели

# In[25]:


x_tr, x_val, y_tr, y_val = train_test_split(x_train_final, y_train,  test_size = 0.2, random_state = 1)


# Обучение модели проводилось несколькими различными классификаторами (логистическая регрессия, случайный лес, градиентный бустинг, нейронная сеть с одним скрытым слоем) с различными параметрами.
#  Качество модели оценивалось по метрике accuracy (т.к. в задании требуется максимизировать кол-во правильно классифицированных объектов, то есть TP+TN).
#  Результаты показали, что лучше всего на имеющихся данных показывает себя градиентный бустинг. Параметры обучения подбирались по сетке. Пример обучения градиентного бустинга показан дальше.

# In[26]:


params = {'n_estimators': [50,100,150],
          'max_depth': [6,8,10],
          'colsample_bytree': [0.6,0.8,1]}
xgb_classifier = xgb.XGBClassifier(random_state=1)
grid_search = GridSearchCV(estimator = xgb_classifier, n_jobs = 2, cv = 4, param_grid = params, 
                           scoring = 'accuracy')
grid_search.fit(x_tr, y_tr)
print(grid_search.best_score_, grid_search.best_params_)


# In[27]:


grid_search.best_estimator_.score(x_val, y_val)


# In[28]:


x_test = x_test.drop(['POLICY_END_MONTH', 'POLICY_BEGIN_MONTH'], axis=1)
x_test = x_test.drop(['POLICY_SALES_CHANNEL'], axis=1)
x_test['POLICY_BRANCH'] = [0 if (x == 'Москва') else 1 for x in x_test['POLICY_BRANCH'].values]
x_test['AGE_18-25'] = [1 if (x<25) else 0 for x in x_test['POLICY_MIN_AGE'].values]
x_test['AGE_25-40'] = [1 if (x>=25)and(x<=40) else 0 for x in x_test['POLICY_MIN_AGE'].values]
x_test['AGE_40-65'] = [1 if (x>=40)and(x<65) else 0 for x in x_test['POLICY_MIN_AGE'].values]
x_test['AGE_65+'] = [1 if (x>=65) else 0 for x in x_test['POLICY_MIN_AGE'].values]
x_test = x_test.drop(['POLICY_MIN_AGE'], axis=1)
x_test['DRV_EXP_0-3'] = [1 if (x<3) else 0 for x in x_test['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_test['DRV_EXP_3-10'] = [1 if (x>=3) & (x<10) else 0 for x in x_test['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_test['DRV_EXP_10-20'] = [1 if (x>=10) & (x<20) else 0 for x in x_test['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_test['DRV_EXP_20+'] = [1 if (x>=20) else 0 for x in x_test['POLICY_MIN_DRIVING_EXPERIENCE'].values]
x_test = x_test.drop(['POLICY_MIN_DRIVING_EXPERIENCE'], axis=1)
x_test = x_test.drop(['POLICY_INTERMEDIARY'], axis=1)
x_test['INSURER_GENDER'] = [0 if (x == 'M') else 1 for x in x_test['INSURER_GENDER'].values]
x_test['POLICY_CLM_N'] = ['N' if (x=='n/d') else x for x in x_test['POLICY_CLM_N'].values]
x_test['POLICY_CLM_GLT_N'] = ['N' if (x=='n/d') else x for x in x_test['POLICY_CLM_GLT_N'].values]
x_test['POLICY_YEARS_RENEWED_N'] = [1 if x=='N' else int(x) for x in x_test['POLICY_YEARS_RENEWED_N'].values]
x_test_numeric = min_max_scaler.transform(x_test[numeric_features])
x_test_binary = x_test[binary_features]
x_test_num_category = one_hot_encoder.transform(x_test[category_num_features])
x_test_cat_CLM_N = label_binarizer.transform(x_test['POLICY_CLM_N'])
x_test_cat_CLM_GLT_N = label_binarizer.transform(x_test['POLICY_CLM_GLT_N'])
x_test_cat_PRV_CLM_N = label_binarizer.transform(x_test['POLICY_PRV_CLM_N'])
x_test_cat_PRV_CLM_GLT_N = label_binarizer.transform(x_test['POLICY_PRV_CLM_GLT_N'])
x_test_cat_CR_REG = label_binarizer_region.transform(x_test['CLIENT_REGISTRATION_REGION'])
x_test_cat_make = label_binarizer_make.transform(x_test['VEHICLE_MAKE'])
x_test_cat_model = label_binarizer_model.transform(x_test['VEHICLE_MODEL'])
x_test_final = sparse.hstack((x_test_numeric, x_test_binary, x_test_num_category, x_test_cat_CLM_N, 
                              x_test_cat_CLM_GLT_N, x_test_cat_PRV_CLM_N, 
                              x_test_cat_PRV_CLM_GLT_N, x_test_cat_CR_REG, x_test_cat_make, x_test_cat_model))


# In[29]:


result_probs = grid_search.best_estimator_.predict_proba(x_test_final)
result_preds = grid_search.best_estimator_.predict(x_test_final)


# In[30]:


results_df = pd.DataFrame()
results_df['POLICY_ID']=x_test.index
results_df['POLICY_IS_RENEWED'] = result_preds
results_df['POLICY_IS_RENEWED_PROBABILITY'] = result_probs[:,1]


# In[34]:


results_df.head(20)


# In[35]:


results_df.to_csv('D:/results.csv', sep=',')


# Полученная в итоге модель - лучшая из всех, с которыми были проведены эксперименты.
#  Улучшить качество теоретически можно при построении модели по сетке с большим числом и разбросом гиперпараметров.
#  Другой способ – изменение качественного и количественного состава участвующих в обучении признаков.
