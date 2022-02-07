#Autocorrelation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#1-Чтение данных
bikes=pd.read_csv("hour.csv")

#2-Предварительный анализ и выбор признаков
bikes_prep=bikes.copy()
bikes_prep=bikes_prep.drop(['index','date','casual','registered'], axis=1)

#3-Чекаем есть ли пропущенные значения
bikes_prep.isnull().sum()

#         4-Визуализация
#4-1-Визуализируем дискретные переменные гистограммой
plt.figure()
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()
plt.show()

#4-2- Визуализация непрерывных переменных
plt.figure()
plt.subplot(2,2,1)
plt.title("Температура Vs Спрос")
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='g')

plt.subplot(2,2,2)
plt.title("Ср.Температура Vs Спрос")
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2, c='b')

plt.subplot(2,2,3)
plt.title("влажность Vs Спрос")
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2, c='m')

plt.subplot(2,2,4)
plt.title("Vветра Vs Спрос")
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2, c='c')

plt.tight_layout()
plt.show()

#4-3-Визуализируем категориальные переменные
#в чем идея: узнать какие категории бывают среди них 
# посчитать средние значения и построить график
colors=['g','b','r','m']
plt.figure()

plt.subplot(3,3,1)
plt.title('Общий спрос в сезон')
cat_list=bikes_prep['season'].unique()
cat_average=bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,2)
plt.title('Общий спрос в месяц')
cat_list=bikes_prep['month'].unique()
cat_average=bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,3)
plt.title('Общий спрос в праздник')
cat_list=bikes_prep['holiday'].unique()
cat_average=bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,4)
plt.title('Общий спрос в выходные')
cat_list=bikes_prep['weekday'].unique()
cat_average=bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,5)
plt.title('Общий спрос в год')
cat_list=bikes_prep['year'].unique()
cat_average=bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,6)
plt.title('Общий спрос по часам')
cat_list=bikes_prep['hour'].unique()
cat_average=bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,7)
plt.title('Общий спрос в раб. дни')
cat_list=bikes_prep['workingday'].unique()
cat_average=bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,8)
plt.title('Общий спрос в раз. погоду')
cat_list=bikes_prep['weather'].unique()
cat_average=bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)
plt.tight_layout()
plt.show()

# Можно сделать вывод, что Распределение спроса зависит от сезона, 
#месяца, праздников, времени суток, погоды

# 5- Чекаем выбросы
bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05,0.1, 0.15, 0.9, 0.95, 0.99])
#вообще не поняла, что произошло

# 6- Чекаем предположения множественной линейной регрессии
correlation= bikes_prep[['temp','atemp','humidity','windspeed', 'demand']].corr()

bikes_prep=bikes_prep.drop(['weekday','year','workingday','atemp','windspeed'], axis=1)
# Чекаем автокорреляцию спроса с помощью accor
df1=pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1, maxlags=12)
#значения спроса довольно сильно связаны со своими прошлыми значениями- автокорреляция

# 7- Изменяем/дополняем признаки 
# нормализуем распределение demand

df1=bikes_prep['demand']
df2=np.log(df1)
plt.figure()
df1.hist(rwidth=0.9, bins=20)
plt.figure()
df2.hist(rwidth=0.9, bins=20)

bikes_prep['demand']=np.log(bikes_prep['demand']) 
#в итоге мы сделали распределение +-нормальным и перевели значения в логарифме в датафрейме
#чекаем автокорреляцию в demand column
t_1=bikes_prep['demand'].shift(+1).to_frame()
t_1.columns=['t-1']
t_2=bikes_prep['demand'].shift(+2).to_frame()
t_2.columns=['t-2']
t_3=bikes_prep['demand'].shift(+3).to_frame()
t_3.columns=['t-3']

bikes_prep_lags=pd.concat([bikes_prep, t_1, t_2, t_3], axis=1)
bikes_prep_lags=bikes_prep_lags.dropna()

#8-create dummy variables and drop first to avoid dummy variables trap-как на русском не понятно

bikes_prep_lags['season']=bikes_prep_lags['season'].astype('category')
bikes_prep_lags['holiday']=bikes_prep_lags['holiday'].astype('category')
bikes_prep_lags['weather']=bikes_prep_lags['weather'].astype('category')
bikes_prep_lags['month']=bikes_prep_lags['month'].astype('category')
bikes_prep_lags['hour']=bikes_prep_lags['hour'].astype('category')

dummy_df=pd.get_dummies(bikes_prep_lags, drop_first=True)

# 9- Разбиваем данные на тестовую и тренир. выборки    
X=bikes_prep_lags[['demand']]
Y=bikes_prep_lags.drop(['demand'], axis=1)
tr_size= 0.7* len(X)
tr_size=int(tr_size)

X_train=X.values[0:tr_size]
X_tes=X.values[tr_size:len(X)]

Y_train=Y.values[0:tr_size]
Y_test=Y.values[tr_size:len(Y)]

# 10- Учим и подгоняем модель
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test= \
    train_test_split(X,Y, test_size=0.4, random_state=1234)