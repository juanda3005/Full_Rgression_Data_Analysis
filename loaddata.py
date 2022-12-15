import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression#regresion lineal de una o varias variables y pipelines
from sklearn.preprocessing import PolynomialFeatures#para polinomios de multiples variables y pipelines
from sklearn.preprocessing import StandardScaler#para normalizar polinomio de multiples vairables y pipelines
#To construct Pipeline
from sklearn.pipeline import Pipeline
from sklearn. metrics import mean_squared_error#To calculate  Mean Squared Error
from sklearn.metrics import r2_score#calcular r2 en modelos polinomiales
from sklearn.model_selection import train_test_split#Para hacer el split de los datos in testing y trainning
from sklearn.model_selection import cross_val_score#para hacer la validación cruada, cross validation
from sklearn.model_selection import cross_val_predict#Predecir con cross validation
from sklearn.linear_model import Ridge #Para realizar la ridge regression
from tqdm import tqdm #Para crear una barra de progreso
from sklearn.model_selection import GridSearchCV #Para realizar el grid search


#pip install -U scikit-learn
# python -m pip show scikit-learn  # to see which version and where scikit-learn is installed
# python -m pip freeze  # to see all packages installed in the active virtualenv
# python -c "import sklearn; sklearn.show_versions()"

#py -m pip install modulo_a_instalar
#py -m pip install jupyter
#py -m pip install notebook
# accedo a clase asi  cd C:\Users\clase
#py -m notebook
#jupyter notebook


#https://priyadogra.com/data-analysis-with-python-cognitive-class-answers/
#https://sparkbyexamples.com/pandas/pandas-convert-string-to-float-type-dataframe/#:~:text=pandas%20Convert%20String%20to%20Float,float64%20%2C%20numpy.
url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df=pd.read_csv(url, header=None)
#read csv asume que los datos contienen un encabezado, al leerlo automaticamente
#se crea un dataframe, no es necesario usar la funcion df=pd.DataFrame(nombredic), si fuera un diccionario si seria
#necesario
#el archivo que voy a cargar no los tiene, asi que tengo que especificarlo
#al poner none automaticante se asigna como header numeros enteros de 0 a n

print("data frame superior",df.head(4))# muestra las primeras filas segun lo que ponga en el arguemnto
#esta funcion es utli para visualizar el df y hacerle posibles cambios
#cargar el dataframe completo consumiria muchos recursos si solo es para visualizar

print("data frame inferior",df.tail(4))#muestra las filas inferiores del dataframe

#asigno los headers correctos al data frame ya que trabajar con numeros entreos puede ser tedioso

headers=["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",\
"body-style","drive-wheels","engine-location","wheel-base","length","width",\
"height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system",\
"bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

df.columns=headers#asigno los nuevos headers

print(df.head(5))

print(len(headers))


path=r"C:\Users\clase\Pictures\Screenshots\automobile.csv"#ruta absoluta, poner la r para omitir los \
# y python no se confunda ya que \ lo toma como ruta de escape ej (\n), o cambiar los \ por /
#asegurarse de poner el .csv en el nombre que le quiero asignar

path2="exported_files/automobile.csv"#ruta relativa, 
#NO PONER COMO NOMBRE A LA CARPETA DESTINO LA EXTENSION DEL ARCHIVO EJ CSV,SQL.

# si no uso una ruta solo pongo el nombre del archivo
# como arumento entre comillas y lo guarda en la carpeta donde esta ubicada 
# la consola df.to_csv('automobile.csv)
df.to_csv(path2,index='False')#el False es para que no se pongan los nombres de las filas

#MIRAR EL TIPO DE DE DATOS QUE POSEE CADA CULMNA(importantes para saber
# que funcion usar en cada columna)
df.dtypes


#MOSTRAR DATOS ESTADISTICOS DE CADA CULUMNA DE MI  DATAFRAME

#(se salta las filas que no contienen datos numericos)
print(df.describe())#no muestra las columnas que no poseen datos numericos
#count:cuenta cuantos valores tiene la culumna

#pero es posible que mi metodo describe funcione tambien para los valores que son tipo objeto
print(df.describe(include="all"))#muestra la estadistica de todas las columnas
#unique:numero de objetos distintos en la columna
#top:el objeto que mas se repite
#freq:es el numero de veces que aprece el objeto que mas se repite
#donde parece NaN es porque esa estadistica en especifico no s epuede calcular para el tipo de datos
#que tiene esa columna

#MOSTRAR INFORMACIÓN DEL TIPO DE DATOS DE LAS PRIMERAS 30 COLUMNASDF(UTIL SI EL DF TIENE MUCHAS COLUMNAS)
print(df.info())






#*********** 1. DATA WRANGLING************************










#ACCEDER A DATOS ESPECIFICOS DEL DF

#COLUMNAS
df["symboling"]
df['symboling']+1 #le suma 1 a todos los valos de la columna 1
df['body-style']




#*************DEAL WITH MISSING DATA*****************








#PUEDEN APARECER COMO ?,NnN,0 o una celda en blanco


#*****************IDENTIFY MISSING VALUES*******



#REEMPLAZAR LOS VALORES DE ? por NaN
df.replace("?", np.nan, inplace = True)
print('DF despues de reemplazar ? por nan','\n',df.head())




#************Evaluating for Missing Data********





#The missing values are converted by default. We use the following functions to identify 
#these missing values. There are two methods to detect missing data:

missing_data = df.isnull()#Muestra true si hay valores nulos en df True si son nulos y False si son validos
#si uso notnull() True si son validosy False si son missing
print('DF de Missing Data','\n',missing_data.head())


# CUENTO CUANTOS VALORES NULOS TIENE CADA COLUMNA
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())#este metodo cuenta el numero de True en la columna:es decir missing data
    print("")  



"""#Filtrar un valor del df
#at:lo uso para pambiar un solo dato usando la etiqueta del renglon o columna
print(df.at[0,'symboling'])

#iat:Cambia un solo dato usando numeros enteros o indices de la fila y columna
print(df.iat[0,0])

#loc:util para filtrar multiples datos del df utilizando la etiqueta
print(df.loc[0:2,'symboling'])

#loc:util para filtrar multiples datos del df utilizando numeros enteros
print(df.iloc[0:2,0])"""


#****************DEAL WITH MISSING DATA*****************


#REEMPLAZAR LOS VALORES FALTANTES (nan) DE UNA COLUMNA POR LA MEDIA DE LA COLUMNA



#Calculo la media de la columna normalizedlosses
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

#Reemplazo los nan por la media normalizedlosses
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


#Calculo la media de la columna bore
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

#Reemplazo los nan por la media bore
df["bore"].replace(np.nan, avg_bore, inplace=True)


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#Otra forma
#df['normalized-losses']=df['normalized-losses'].replace(np.nan,df['normalized-losses'].mean)
#print('valores nan reemplazados por la media ','\n',df)


#REEMPLAZAR LOS VALORES FALTANTES DE UNA COLUMNA DE STRINGS O CARACTERES



print(df['num-of-doors'].value_counts())#me muestra los valores que hay y cuantas veces se repiten
frecuent_num_doors=df['num-of-doors'].value_counts().idxmax()#idmax() identifica el valor que mas se repite

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, frecuent_num_doors, inplace=True)


# Elimina todos los valores con NaN de la columna "precio"
df.dropna(subset=["price"], axis=0, inplace=True)

# restablece el índice por defecto, porque se eliminaron dos filas, se eliminaron 4 registros que tenian NaN
df.reset_index(drop=True, inplace=True)
print(df['price'])

print('Dataframe despues de corregir eliminar los Nan de price  y corregir las demas columnas','\n',df.head())


#IDENTIFICAR ESPACIOS EN BLANCO
df= df.replace(" ",'@',regex=True)
print('identificar espacios en blanco con un @','\n',df)

#Borrar espacios en blanco al inicio y al final en todo el df
df=df.replace(r"^ +| +$", r"", regex=True)
check=df.replace(" ",'@',regex=True)




#********* CORRECT DATA FORMAT -- FORMATING DATA*******





#CAMBIO EL FORMATO DE LAS COLUMNAS AUN FORMATO ADEUCADO PARA SUS VALORES
#ALGUNOS SE CAMBIARON ANTERIORMENTE PARA CALCULAR SU MEDIO, AQUI  SE RESTABLECEN
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df[["horsepower"]] = df[["horsepower"]].astype("int")

print('DF luego de corregir el formato de los datos','\n',df)

#CONVERTIR EL TIPO DE DATOS DE UNA COLUMNA CATEGORICA A NUMERICA SI APARECE ERROR

#PRICE SON VALORES NUMERICOS PERO ME SALE OBJECT, LO CAMBIO
#TIENE QUE SER EN ESTE ORDEN
df['price'] = pd.to_numeric(df['price'], errors='coerce')#esto lo converte a float, luego lo paso a int
df['price'] = df['price'].replace(np.nan, 0, regex=True)#reemplazo los NaN por 0
df['price'] = df['price'].astype(float)
print(df['price'].dtypes)


#****************ESTANDARIZACIÓN DE DATOS******************




#Aplicar calculos a toda una columna

#convertir mpg a L/100km
df['city-mpg'] = 235/df['city-mpg']

df.rename(columns={"city-mpg":"city-L/100km"}, inplace = True)
print('columna city-L calculada','\n',df['city-L/100km'])




#*************+NORMALIZACIÓN DE DATOS*********************




#AYUDA A VOLVER CONSISTENTES LOS RANGOS DE LAS VARIABLES PARA SU POSTERIOR ANALISIS

# La normalización es el proceso de transformar valores de varias variables en un rango similar. 
# Las normalizaciones típicas incluyen escalar la variable para que el promedio de la variable sea 0, 
# escalar la variable para que la varianza sea 
# 1 o escalar la variable para que los valores de la variable oscilen entre 0 y 1.

#Simple Feature scaling
df["length"] = df["length"]/df["length"].max()

#Min-Max
df["length"]=(df["length"]-df["length"].min() )/(df["length"].max()-df["length"].min())

#Z-score in python
df['length']=(df['length']-df['length'].mean())/df['length'].std()






#****************************BINNING EN PYTHON (O AGRUPAR)***************





# El agrupamiento es un proceso de transformación de variables 
# numéricas continuas en 'contenedores' categóricos discretos para el análisis agrupado.

#AQUI DIVIDO A LA CULUMNA PRICE EN INTERVALOS (LOW,MEDIUM,HIGH)

#EL AGRUPAMIENTO AQUI ES MUY UTIL PORQUE LOGRO REDUCIR 59 CATEGORIAS A 3

bins = np.linspace(min(df["price"]), max(df["price"]), 4)#cuantro numeros espaciados de igual forma
#me da como resultado 3 intervalos con el mismo ancho I-------I--------I---------I(start,end,numeros_generados)
group_names = ["Low", "Medium", "High"]#asigno el nombre de los intervalos
df['price-binned']= pd.cut(df['price'], bins, labels=group_names, include_lowest=True)
#creo los bins o contenedores en pandas y los asigno a una nueva columna que se agrega al df

print('Binned group price')
print(df[['price','price-binned']].head(20))
print('Numero de vehiculos en price binned:',df["price-binned"].value_counts())#veo cuantos vehiculos
#hay en cada cetegoria que creé

#GRAFICO UN HISTOGRAMA DE LA AGRUPACIÓN DE (PRICE-BINNED)
plt.bar(group_names, df["price-binned"].value_counts())

# set x/y labels and plot title
plt.xlabel("price")
plt.ylabel("count")
plt.title("price bins")

# plt.show()







#************************+TURNING CATEGORICAL VARIABLES INTO CUANTITIVE VARIABLES**********
#UNA VARIABLE CATEGORICA PUEDO CAMBiARSE A NUMERICA CONTANDO SI APARECE O NO CON 1 Ó 0




print(df.columns)
dummy_variable_1 = pd.get_dummies(df["fuel-type"])#creo las columnas dummy, puedo poner un prefijo(df['fuel-type'], prefix='dummy') 

#le cambio el nombre a las columnas dummy para mayor claridad o uso el prefijo en la funcion anterior
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)

df=pd.concat([df,dummy_variable_1], axis = 1)#agrego las columnas dummy al df
df=df.drop(['fuel-type'], axis = 1)#si quiero borrar la columna categorica original del df
print("VARIABLES CATEGORICAL DE fuel-type convertidas a dummy, esto srive para verificar\
      a que variable se le asignó el 1 o el 0, si son dos solo dos categorias a separar se puede poner en una misma\
      columna, si hay mas de dos se deben separar, aqui se separan dos pero es solo por el ejemplo\
          \
    ¿COMO ASIGNA EL 0 O 1?: EN fuel-type-gas A LA CELDA QUE CONTIENE GAS LE PONE 1, A LA CELDA QUE CORRESPONDE A OTRA CATEGORIA LE PONE 0\
        Y LO MISMO CON LAS DEMAS COLUMNAS O FEATURES\
    SI ESTUVIERAN EN LA MISMA COLUMNA SE DEBE VERIFICAR A QUE CATEGORIA SE LE PUSO 1 Y A CUAL 0\
    ES BASTANTE UTIL SABERLO A LA HORA DE ANALIZAR UN ARBOL DE DECISION")
print(df[['symboling','fuel-type-gas','fuel-type-diesel','price','price-binned']])







#******************************** 2.EXPLORATORY DATA ANALYSIS****************







#******************How to choose the right visualization method?


#Debo observar el tipo de variables que tiene el df para escoger el metodo de visualización adecuado
print('Observación del tipo de variables para su escoger su visualización ','\n',df.dtypes)

#******************* OBSERVAR LA CANTIDAD Y TIPO DE DATOS DE UNA COLUMNA*****++

print(df[ "drive-wheels"].value_counts())


#********************************CORRELATION*********************


#For example, we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":

print('Correlación entre las variables (entre int y float (solo se hace entre numericas)','\n',df.corr())


#*******************************VARIABLES CONTINUAS Y REGRESIÓN LINEAL*************





#Continuous numerical variables are variables that may contain any value within some range. 
# They can be of type "int64" or "float64". 
# A great way to visualize these variables is by using scatterplots with fitted lines.

#"regplot" TRAZA EL DIAGRAMA DE DISPERSIÓN MAS LA LINEA D EREGRESIÓN AJUSTADA A LOS DATOS

#************************************** SCATTER PLOT******************************************


# Engine size as potential predictor variable of price

sns.regplot(x="engine-size", y="price", data=df)
plt.title("Scatter plot (price vs engine-size) and Linear Regresion ")
plt.ylim(0,)
# plt.show()

#Analysis:As the engine-size goes up, the price goes up: this indicates a positive direct correlation 
# between these two variables. Engine size seems like a pretty 
# good predictor of price since the regression line is almost a perfect diagonal line.

#Se observa la correlación entre estas dos variables
print('correlación engine-size y price','\n',df[["engine-size", "price"]].corr())
#La correlación es aporximadamente de 0.87(Positive linear relationship)


# peak-rpm as potential predictor variable of price

sns.regplot(x="peak-rpm", y="price", data=df)
plt.title("Scatter plot (peak-rpm vs engine-size) and Linear Regresion ")
plt.ylim(0,)
# plt.show()

#Analysis:Peak rpm does not seem like a good predictor of the price at all since 
# the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted 
# line, showing lots of variability. Therefore, it's not a reliable variable.

#Se observa la correlación entre estas dos variables
print('correlación peak-rpm y price','\n',df[["peak-rpm", "price"]].corr())
#La correlación es aporximadamente de -0.101616 (Weak linear relationship)




#***********************************+VARIABLES CATEGORICAS********************








#These are variables that describe a 'characteristic' of a data unit, and are selected 
# from a small group of categories. The categorical variables can have the type "object" or "int64". 
# A good way to visualize categorical variables is by using boxplots.


#*************** BOX PLOTS***********************************

#BODY-STYLE TO PREDICT  PRICE

#Upper quartile: Representa el percentile 75
#Lower quartile: Representa el percentile 25
#Upper extreme: 1.5 veces el Rango Interquartile (IQR=upper quartile-lower quartile)

sns.boxplot(x="body-style", y='price', data=df)
#Analysis:We see that the distributions of price between the different 
# body-style categories have a significant overlap, so body-style would not be a 
# good predictor of price. Let's examine engine "engine-location" and "price":

#ENGINE-LOCATION TO PREDICT PRICE

sns.boxplot(x="engine-location", y='price', data=df)
#Analysis:Here we see that the distribution of price between these two engine-location categories, front and rear, 
# are distinct enough to take engine-location as a potential good predictor of price.


#DRIVE-WHEEL TO PREDICT PRICE
sns.boxplot(x='drive-wheels', y='price',data=df)
#Analysis;Here we see that the distribution of price between the different drive-wheels categories differs. 
# As such, drive-wheels could potentially be a predictor of price.



#***************************ESTADISTICA DEXCRIPTIVA***************************




#Describe method:

# This will show:

# the count of that variable
# the mean
# the standard deviation (std)
# the minimum value
# the IQR (Interquartile Range: 25%, 50% and 75%)
# the maximum value
# We can apply the method "describe" as follows:

print('resumen estadistico luego del data wranglin (no incluye tipo object)','\n',df.describe())

print('resumen estadistico luego del data wranglin(todos los tipo de datos)',df.describe(include="all"))
#muestra la estadistica de todas las columnas

# The default setting of "describe" skips variables of type object. 
# We can apply the method "describe" on the variables of type 'object' as follows:
print('resumen estadistico luego del data wranglin de tipo object','\n',df.describe(include=['object']))



#********VALUE COUNTS************

df['drive-wheels'].value_counts()#Muestra las valores y la cantidad que estos se repiten en la columna
# Don’t forget the method "value_counts" only works on pandas series, not pandas dataframes. 
# As a result, we only include one bracket df['drive-wheels'], 
# not two brackets df[['drive-wheels']].

#La serie se puede convertir a dataframe
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
#renombro la columna donde sale el valor

drive_wheels_counts.index.name = 'drive-wheels'#le pongo nombre a la columna de la izquierda

print('value.counts convertido a dataframe','\n',drive_wheels_counts)


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

#En el caso de esta variable no sirve como predictor ya que front location tiene 198 y rear solo 3,
#aunque hay una diferencia entre ambos valores este valor esta sesgado y no se pueden sacar conclusiones acerca
#de engine location para predecir price



  

#************ GROUP BY, PIVOT TABLE  AND HEAT MAP ***********************************


df_group_one = df[['drive-wheels','body-style','price']]

#AGRUPO UNA VARIABLE CON LA VARIABLE OBJETIVO

#Utilizo la funcion group by para agrupar con la ultima variable que puse en la lista en df_group_one

df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
#Obtengo una tabla con drive-wheels y price
#Analysis:From our data, it seems rear-wheel drive vehicles are, on average, the most expensive, 
# while 4-wheel and front-wheel are approximately the same in price.


#AGRUPO VARIAS VARIABLES CON LA VARIABLE OBJETIVO

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)
#se muestra el precio promedio por cada categoria
#es decir resume  todas las categorias de las dos variables y saca el precio promedio que estas tienen


#CONVERTIR LAS VARIABLES AGRUPADAS A UNA TABLA PIVOTE 

#Pero para visualizar mejor los datos utiliza la función pivot y convierto a una de las variables a fila
#para poder observar una mejor relación de estas variables con el precio, igual que excel spreddshets

#PIVOT TABLE

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')#pongo a drive-wheels como fila

#En ocasiones no aparecen valores en algunas de las celdas, estas se pueden llenar con el valor 0 pero se puede
#poner cualquiera de los valores mencionados en la sección de data wrangling
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print('pivot table drive-wheels body-style with price','\n',grouped_pivot)

#HEAT MAP
#Grafica simple sin etiquetar

# plt.pcolor(grouped_pivot, cmap='RdBu')
# plt.colorbar()
# # plt.show()

#Grafica completa con ETIQUETADO DE LAS CATEGORIAS DE LAS COLUMNAS

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
# plt.show()



#*******************CORRELATION AND CAUSATION*************************





# Correlation: a measure of the extent of interdependence between variables.

# Causation: the relationship between cause and effect between two variables.

#It is important to know the difference between these two. Correlation does not imply causation. 
# Determining correlation is much simpler the determining 
# causation as causation may require independent experimentation.



#*******************CORRELACIÓN DE PEARSON***********




# The Pearson Correlation measures the linear dependence between two variables X and Y.

# The resulting coefficient is a value between -1 and 1 inclusive, where:

# 1: Perfect positive linear correlation.
# 0: No linear correlation, the two variables most likely do not affect each other.
# -1: Perfect negative linear correlation.
# Pearson Correlation is the default method of the function "corr". Like before, we can 
# calculate the Pearson Correlation of the of the 'int64' or 'float64' variables.

print("correlación de pearson ( la que hace por defecto python entre int y los float",'\n',df.corr())


#P-VALUE

#  The P-value is the probability 
#  value that the correlation between these two variables is 
#  statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% 
#  confident that the correlation between the variables is significant.

# p-value is  < 0.001: we say there is strong evidence that the correlation is significant.
# the p-value is  < 0.05: there is moderate evidence that the correlation is significant.
# the p-value is  < 0.1: there is weak evidence that the correlation is significant.
# the p-value is  > 0.1: there is no evidence that the correlation is significant.

#We can obtain this information using "stats" module in the "scipy" library.


#WHEEL-BASE vs PRICE

#Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient between wheel-base and price is", pearson_coef, " with a P-value of P =", p_value) 
#Analysis:Since the p-value is  <0.001, the correlation between wheel-base and price is statistically significant, 
#although the linear relationship isn't extremely strong (~0.585).


# Horsepower vs. Price

# Let's calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'.

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient between horsepower and price is", pearson_coef, " with a P-value of P = ", p_value)  
#Analysis:Since the p-value is  < 0.001, the correlation between horsepower and price is 
#statistically significant, and the linear relationship is quite strong (~0.809, close to 1).


# Length vs. Price

# Let's calculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'.

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient between lenght and price is", pearson_coef, " with a P-value of P = ", p_value)  
#Analysis:Since the p-value is  < 0.001, the correlation between length and price is statistically significant, 
#and the linear relationship is moderately strong (~0.691).


# Width vs. Price

# Let's calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price':
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
#Analysis:Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the 
# linear relationship is quite strong (~0.751).


#Curb-Weight vs. Price

#Let's calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price':
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#Analysis:Since the p-value is  < 0.001, the correlation between curb-weight and price is statistically 
# significant, and the linear relationship is quite strong (~0.834).


# Engine-Size vs. Price

# Let's calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price':
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#Analysis:Since the p-value is  < 0.001, the correlation between engine-size and price is 
# statistically significant, and the linear relationship is very strong (~0.872).


# Bore vs. Price

# Let's calculate the Pearson Correlation Coefficient and P-value of 'bore' and 'price':
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )
#Analysis:Since the p-value is  < 0.001, the correlation between bore and price is statistically significant, 
# but the linear relationship is only moderate (~0.521).


#Highway-mpg vs. Price

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 
#Analysis:Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant
# , and the coefficient of about -0.705 shows that the 
# relationship is negative and moderately strong.

#city-L/100km vs. Price

pearson_coef, p_value = stats.pearsonr(df['city-L/100km'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#Analysis:Since the p-value is  < 0.001, the correlation between city-L/100km and price is 
#statistically significant, and the coefficient of about -0.687 shows that the relationship is 
# negative and moderately strong.





#*****************************++ANOVA METHOD /ANALYSIS OF VARIANCE)****************



# El Análisis de Varianza (ANOVA) es un método estadístico utilizado para probar si existen diferencias
# significativas entre las medias de dos o más grupos. ANOVA devuelve dos parámetros:



#F-score: ANOVA asume que las medias de todos los grupos son iguales, calcula cuánto se 
# desvían las medias reales de la suposición y lo informa como el puntaje de la prueba F. Una puntuación
# mayor significa que hay una diferencia mayor entre las medias.

# Valor P: el valor P indica cuán estadísticamente significativo es nuestro valor de puntaje calculado.

# Si nuestra variable de precio está fuertemente correlacionada con la variable que estamos analizando, 
# esperamos que ANOVA arroje una puntuación de prueba F considerable y un valor de p pequeño.


#F-testLa prueba F calcula la relación de variación entre la media de los grupos 
# sobre la variación dentro de cada uno de los grupos de muestra.
#p-value:El valor p muestra si el resultado obtenido es estadísticamente significativo.
"""NOTA: si F-score is large (F large ej=400 and small p-vale ej=1.05e-5) the 
correlation is strong(variación grande entre la media de los grupos analizados
significa correlación fuerte con la variable objetivo)

Small F-score (F less than 1, pvalue larger than 0,05) imply poor correlation between variable 
categories and target variable(variación baja entre la media de los grupos analizados
significa correlación baja con la variable objetivo)"""

#PARA ANALIZAR EL EFECTO DE UNA VARIABLE CATEGORICA COMO drive-wheels en EL TARGET:PRICE

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
#anteriormente ya se habia definido el df df_gptest, solo se toman dos de sus columnas y se usa groupby

#Utilizo get group solo para extraer la columna de precios del grupo creado (se extraen los promedios de las
# categorias de las dos variables con respecto al precio)
grouped_test2.get_group('4wd')['price']

#We can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.

# ANOVA
#Selecciono con get_group las categorias de drive-wheels junto con la columna price
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results drive-wheels and price: F=", f_val, ", P =", p_val)   


# This is a great result with a large F-test score showing a strong correlation 
# and a P-value of almost 0 implying almost certain statistical significance. But does this mean 
# all three tested groups are all this highly correlated?

# Let's examine them separately.
#fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results fwd and rwd: F=", f_val, ", P =", p_val )

#4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)   


#4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)   


#************************Conclusion: Important Variables****************




# We now have a better idea of what our data looks like and which variables are 
# important to take into account when predicting the car price. We have narrowed it down to the following variables:

# Continuous numerical variables:

# Length
# Width
# Curb-weight
# Engine-size
# Horsepower
# City-mpg
# Highway-mpg
# Wheel-base
# Bore

# Categorical variables:
# Drive-wheels


# As we now move into building machine learning models to automate our analysis, 
# feeding the model with variables that meaningfully affect our target variable will improve our 
# model's prediction performance.







#**********************************************3. DESARROLLO DEL MODELO*********************************


#****************************LINEAR AND MULTIPLE LINEAR REGRESSION******


#*******************SIMPLE REGRESSION******

# Linear Regression
# One example of a Data Model that we will be using is:

# Simple Linear Regression
# Simple Linear Regression is a method to help us understand the relationship between two variables:

# The predictor/independent variable (X)
# The response/dependent variable (that we want to predict)(Y)
# The result of Linear Regression is a linear function that predicts the response (dependent) variable as 
# a function of the predictor (independent) variable.

#Y: Response 
#X: Predictor 

#Yhat = a + b  X
# a refers to the intercept of the regression line, in other words: the value of Y when X is 0
# b refers to the slope of the regression line, in other words: the value with which Y changes 
# when X increases by 1 unit


#CREO EL OBJETO DE REGRESIÓN LINEAL
lm=LinearRegression()

# How could "highway-mpg" help us predict car price?

#For this example, we want to look at how highway-mpg can help us predict car price. 
# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)#Ajusto el modleo lineal

#We can obtain the prediction
Yhat=lm.predict(X)#obtengo un arreglo con los valores de y de la muestra predecidos, esta operación 
#calcula el valor del precio con la ecuación de regression con los x de la base de datos o muestra

print("Primeros 5 valores de Y predecidos obtenidos segun los X de la muestra tomada",Yhat[0:5])


#What is the value of the intercept (a)?
print('Intercepto ',lm.intercept_)

#What is the value of the slope (b)
print('Pendiente',lm.coef_)

#What is the final estimated linear model we get?
"Price = 38423.31 - 821.73 x highway-mpg"
#si highway-mpg sube un galon el precio disminuye 821.73




# How could "engine-size" help us predict car price?

#For this example, we want to look at how highway-mpg can help us predict car price. 
# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.

lm1=LinearRegression()

Xa = df[['engine-size']]
Ya = df['price']

lm1.fit(Xa,Ya)#Ajusto el modleo lineal

#We can obtain the prediction
Yhat1=lm1.predict(Xa)#obtengo un arreglo con los valores de y de la muestra

#What is the value of the intercept (a)?
print('Intercepto ',lm1.intercept_)

#What is the value of the slope (b)
print('Pendiente',lm1.coef_)

#What is the final estimated linear model we get?
"Price=-7963.34 + 166.86*engine-size"
"Yhat=-7963.34 + 166.86*X"




#**********************++MULTIPLE LINEAR REGRESSION*****************************+




# From the previous section we know that other good predictors of price could be:

# Horsepower
# Curb-weight
# Engine-size
# Highway-mpg
# Let's develop a model using these variables as the predictor variables.

Z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
lmm=LinearRegression()

#Fit the linear model using the four above-mentioned variables.
lmm.fit(Z,df['price'])
#aqui se usó el objeto de la regression lineal simple, pero mientras se usen predictores y variables a predecir
#diferentes no saca error

#What is the value of the intercept(a)?
print("Intercepto de la regresion lineal multiple",lmm.intercept_)

#What are the values of the coefficients (b1, b2, b3, b4)?
print("coeficientes de la regresion lineal multiple",lmm.coef_)

"""Using the function:Yhat = a + b1*X1 + b2*X2 + b3*X3 + b4*X4
Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg

Los interceptos se muestran por orden según como se pusieron en el lm.fit
"""


#REGRESION MULTIPLE USING NORMALIZE-LOSSES AND HIGHWAY-MPG


lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
print("Intercepto de la regresion lineal multiple normalize-losses and highwway",lm2.intercept_)
print("Pendiente de la regresion lineal multiple normalize-losses and highwway",lm2.coef_)











#**************************MODEL EVALUATION USING VISUALIZATION**************




#**** SIMPLY REGRESION PLOT

# This plot will show a combination of a scattered data points (a scatterplot), as well 
# as the fitted linear regression line going through the data. This will give us a reasonable 
# estimate of the relationship between the two variables, the strength of the 
# correlation, as well as the direction (positive or negative correlation).

#Let's visualize highway-mpg as potential predictor variable of price:

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.xlabel('highway-mpg')
plt.ylabel('price')
# plt.show()
plt.ylim(0,)
#Analysis: L linea de regressión muestra na pendiente de regresión negativa
#sin embargo esto no es suficiente para determinar si el modelo lineal es el correcto para realizar
#una predicción con esta variable, se debe verificar que tan esparcidos estan los puntos 
# alrededor de la linea de regresión, si estan muy separados el modelo lineal puede no ser el adecuado.


#Let's compare this plot to the regression plot of "peak-rpm".
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.xlabel('peak-rpm')
plt.ylabel('price')
# plt.show()
plt.ylim(0,)

#Analysis:Comparing the regression plot of "peak-rpm" and "highway-mpg", we see that the points for 
# "highway-mpg" are much closer to the generated line and, on average, decrease. The points for "peak-rpm" 
# have more spread around the predicted line and it is much harder to determine if the points are
#  decreasing or increasing as the "peak-rpm" increase


#Lo verificamos usando el método de correlación
print('Correlación entre highway-mpg, peak-rpm y price','\n',df[["peak-rpm","highway-mpg","price"]].corr())




#************+RESIDUAL PLOT TO VALIDATE SIMPLY LINEAR REGRESSION MODEL*******




#COMO SE PUEDE VISUALIZAR LA VARIANZA DE LOS DATOS QUE SE VIERON EN EL GRAFICO DE REGRESIÓN?

#Cuando es bueno un residual plot para validar completamente el modelo de regresión lineal?

#1.Es correcto si los datos estan esparcidos de forma aleatoria alrededor del eje x
#NO DEBE HABER NINGUN PATRÓN RECONOCIBLE EN LOS DATOS.

#No es correcto si obseervo que los valores aumentan o disminuyen alrededor del eje
#X(Cambia la varianza alrededor del eje x), es decir no puede haber ningun patrón en los datos.

#NO es correcto si hay curvatura (parabola den la grafica), esto quiere decir que no estan aleatoriamente
#separados, Esto tambien sugiere que un modelo lineal no se ajusta y puede ser unO No lineal.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
# plt.show()
#Analysis: Se observa que hay un patron de curvatura en los datos, asi que no estan separados aleatoriamente
#alrededor del eje x, asi que un modelo no lineal puede ser el adecuado para esta variable.





#****************************MULTIPLE REGRESSION PLOT


Y_hat = lmm.predict(Z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")#false en hist para que grafique 
#a distribución y no el histograma, aqui pongo los x vs y de la base de datos (valores actuales)

sns.distplot(x=Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
#aqui pongo los x vs (los y predichos con la ecuación)

#price y yhat son valores para precio, en la grfica se grafican ambos en el eje x,
#asi es la grafica de distribución, y el ax es para agregar otra variable a la distrivución
#para poder comparar, en el eje y se pone proporcion de carros pero solo es la densidad de la distribución


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars (Density)')

# plt.show()
plt.close()

#Analysis:We can see that the fitted values are reasonably close to the actual values since the two distributions 
# overlap a bit. However, there is definitely some room for improvement.





#****************************POLINOMIAL REGRESSION AND PIPELINES****************



#**********************************+SIMPLE POLINOMIAL REGRESSION*******************




# Polynomial regression is a particular case of the general linear regression model
# or multiple linear regression models.
# We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.
# There are different orders of polynomial regression:

# Quadratic - 2nd Order
# Yhat = a + b_1 X +b_2 X^2 


# Cubic - 3rd Order
# Yhat = a + b_1 X +b_2 X^2 +b_3 X^3

# Higher-Order
# Y = a + b_1 X +b_2 X^2 +b_3 X^3

# We saw earlier that a linear model did not provide the best fit while using "highway-mpg" as the predictor variable. 
# Let's see if we can try fitting a polynomial model to the data instead.




#***SIMPLY POLINOMIAL REGRESSION

#SIMPLY POLINOMIAL REGRESSION BETWEEN highway-mpg and price

#Realizo una funcion para graficar polinomios y la llamo PlotPolly
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    # plt.show()
    plt.close()

#Defino las variables que quiero graficar de forma polinomica
x=df['highway-mpg']#Aqui polyfit necesita necesita un valor de 1D
y=df['price']

#Let's fit the polynomial using the function polyfit, 
# then use the function poly1d to display the polynomial function.

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print('Polinomio de orden 3 para highwat-mpg y price: ','\n',p)

#Let's plot the function:
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)

#Analysis:We can already see from plotting that this polynomial model performs better than the linear model. 
#This is because the generated polynomial function "hits" more of the data points.

#Que tal si probamos con un polinomio de mayor orden como 11?
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print('Polinomio de orden 11 para highwat-mpg y price: ','\n',p)
PlotPolly(p1,x,y, 'Highway MPG')
#Analysis:En este caso obtuve el mismo polinomio que el de grado 3, asi que dejo el de grado 3





#*************************MULTIPLE POLINOMIAL REGRESSION (MAS DE UNA DIMENSIÓN)*********








#The analytical expression for Multivariate Polynomial function gets complicated. 
# For example, the expression for a second-order (degree=2) 
# polynomial with two variables is given by:

#Yhat = a + b1 X_1 +b_2 X_2 +b_3 X_1 X_2+b_4 X_1^2+b_5 X_2^2


#*********PRE-PROCESAMIENTO O NORMALIZACIÓN DEL POLINOMIO

#As the dimension of the data gets larger we may want to normalize multiple features in scikit-learn
#instead we can use the  preprocessing module to simplify many tasks

#first import
#from sklearn.preprocessing import Standardscaler


SCALE=StandardScaler()#We train the object
SCALE.fit(Z)#fit scaled object
Z_scale=SCALE.transform(Z)#transformo los datos en un nuevo data_frame escalado
print('Z escalado','\n',Z_scale)


#TRANSFORMAR LOS DATOS EN POLINOMIO

#Se puede hacer sin escalar antes los datos originalesy usar el Z original,
#pero se recomienda hacerlo para manejar mejor los datos
#NOTA: SE DEBE ESCAAR PRIMERO ANTES DE HACER LA OPERACIÓN, NO ESCALAR DESPUES YA QUE EL RESULTADO
#ES DIFERENTE


#En este caso polyfit no nos sirve para hallar la ecuación de regresión

#We can perform a polynomial transform on multiple features. First, we import the module:
#from sklearn.preprocessing import PolynomialFeatures

#We create a PolynomialFeatures object of degree 2:
pr=PolynomialFeatures(degree=2)

#Ihen we transform the features into a polynomial feature With the fIt_transform mnethod

Z_pr=pr.fit_transform(Z_scale)#z es la variable que contiene las columnas o features a transformar
print('Transformed features MULTIPLE POLYNOMIAL REGRESSION','\n',Z_pr)
print('In the original data, there are 201 samples and 4 features.',Z.shape)
#In the original data, there are 201 samples and 4 features.

print('After the transformation, there are 201 samples and 15 features.',Z_pr.shape)
#After the transformation, there are 201 samples and 15 features.





#***************************************PIPELINES*********************************************




#HACE LO MISMO QUE LA SLR Y MLR y POLINOMIAL, SOLO QUE DE UNA FORMA MAS CORTA



# Data Pipelines simplify the steps of processing the data. We use the module Pipeline
#  to create a pipeline. We also use StandardScaler as a step in our pipeline.

#CON LAS PIPELINES PUEDO NORMARLIZAR, TRANSFORMAR Y CREAR EL MODELO POR REGRESSION LINEAL
# /POLINOMIAL EN UN SOLO PROCESO

#We create the pipeline by creating a list of tuples 
# including the name of the model or estimator and its corresponding constructor

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
# el ultimo elemento contiene el nombre del estimador:'scale','polynomial','model'
# el segundo elemento contiene el contructor del modelo:StandardScaler,PolynomialFeature,LinearRegression.

#We input the list as an argument to the pipeline constructor:
pipe=Pipeline(Input)#obtengo un pipeline object


#First, we convert the data type Z to type float to avoid conversion warnings that may appear as a 
# result of StandardScaler taking float inputs.
#Then, we can normalize the data, perform a transform and fit the model simultaneously.
Z = Z.astype(float)

print('Ajuste del modelo','\n',pipe.fit(Z,df['price']))

#Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously.
yhatpipe=pipe.predict(Z)
print("Predicción MULTIPLE POLINOMIAL FIT usando pipelines normalizacion,transformacion,regression",'\n',yhatpipe)
#No me muestra una expresión algebraica clara, sin embargo, basta con ingresar en el argumento de predict
#los valores correspondientes a las variables que quiero ingresar para predecir el precio y obtener un resultado


#SIMPLE POLYNOMIAL REGRESSION USING PIPELINES

X = X.astype(float)
pipe=Pipeline(Input)
pipe.fit(X,Y)
ypipe=pipe.predict(X)
print("Predicción SIMPLE POLINOMIAL FIT PIPELINES",'\n',ypipe[0:4])


# SLR USING PIPELINES (OMITO LA TRANSFORMACIÓN Y ME DA IGUAL QUE EL SLR)

Input1=[('scale',StandardScaler()),('model',LinearRegression())]
pipe=Pipeline(Input1)
pipe.fit(X,Y)
ypipe=pipe.predict(X)
print("Predicción SLR con PIPELINES (OMITO TRANFORMACIÓN) ",'\n',ypipe[0:4])

# MLR  USING PIPELINES (OMITO LA TRANSFORMACIÓN Y ME DA IGUAL QUE EL MLR)

pipe=Pipeline(Input1)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print("Predicción MLR con PIPELINES (OMITO TRANFORMACIÓN) ",'\n',ypipe[0:4])






#*****************************4. Measures for In-Sample Evaluation**********





# When evaluating our models, not only do we want to visualize the results, but we also want a quantitative measure to determine how accurate the model is.

# Two very important measures that are often used in Statistics to determine the accuracy of a model are:

# R^2 / R-squared
# Mean Squared Error (MSE)



#R-SQUARED


# R squared, also known as the coefficient of determination, is a measure to indicate how close the data is
#  to the fitted regression line.
# The value of the R-squared is the percentage of variation of the response variable (y) 
# that is explained by a linear model.

#EL R2 ESTA USUALMENTE ENTRE 0 Y 1, SI R2 DA NEGATIVO SE DEBE A UN SOBREAJUSTE
#DEPENDIENDO DEL CAMPO SE PUEDE ACEPTAR O NO EL VALOR R2, ALGUNOS AUTORES
#SUGIEREN UN VALOR >=0.10


#Mean Squared Error (MSE)

#The Mean Squared Error measures the average of the squares of errors. 
# That is, the difference between actual value (y) and the estimated value (ŷ).




#MEDIDAS ENTRE EL MODELO SIMPLE DE REGRESSION LINEAL DE HIGHWAY-MPG Y PRICE





#CALCULO DEL MSE 

#Recordar que hay que calcular el Yhat con predict()
print('MSE entre highway-price en modelo de regresion simple:',mean_squared_error(df['price'],Yhat))
#Toma como argumentos el valor actual y el predicho de la variable objetivo

#CALCULO EL R2 DEL MODELO


#highway_mpg fit, ya se habia calculado antes, no es necesario hacerlo de nuevo, pero siempre es necesario
#calcularlo previamente para el R2, en si fit es el que crea o ajusta el modelo, con predict ya muestro las variables
#necesarias para hacer la predicción
lm.fit(X, Y)
# Find the R^2
print('The R2 entre highway-price en modelo de regresion simple: ', lm.score(X, Y))

#Analysis:R2=0.4965911884339176
#  can say that ~49.659% of the variation 
# of the price is explained by this simple linear model "horsepower_fit".
#esto termina de corroborar la decision anterior de tomar un modelo polinomial
#para predecir price a partir de hihgway-mpg





#*********MEDIDAS PARA EL MODELO MULTIPLE DE REGRESSION LINEAL





#CALCULO DEL MSE 

lmm.fit(Z, df['price'])
Y_predict_multifit = lmm.predict(Z)


print('The mean square CON REGRESION LINEAL MULTIPLE: ', \
      mean_squared_error(df['price'], Y_predict_multifit))
# #Toma como argumentos el valor actual y el predicho de la variable objetivo


# #CALCUO DEL R2

# # fit the model 
# # Find the R^2
print('The R2 entre el modelo de REGRESION MULTIPLE:: ',lmm.score(Z, df['price']))

# #Analysis:We can say that ~80.937 % of the variation 
# of price is explained by this multiple linear regression "multi_fit".




#MEDIDAS PARA EL MODELO POLINOMIAL SIMPLE (DE UNA SOLA VARIABLE JUNTO CON LA VARIABLE OBJETIVO)




# MODELO POLINOMIAL ENTRE HIGHWAY-MPG Y PRICE


#IMPORTAR:
#from sklearn.metrics import r2_score



#CALCULO DE R2

r_squared = r2_score(y, p(x))
print('The R-square para MODELO POLINOMIAL SIMPLE ES: ', r_squared)

#Analysis:We can say that ~67.419 % of the variation of price is explained by this polynomial fit.


#CALCULO MSE
print('The MSE para MODELO POLINOMIAL SIMPLE ES:', mean_squared_error(df['price'], p(x)))






#*********MEDIDAS PARA EL MODELO DE REGRESSION LINEAL MULTIPLE CON PIPELINES





#CALCULO DEL MSE 

pipe=Pipeline(Input)
pipe.fit(Z,df['price'])
Z = Z.astype(float)
yhat_multipoly=pipe.predict(Z)


print('The mean square CON MULTIPLE POLINOMIAL REGRESSION CON PIPELINES: ', \
      mean_squared_error(df['price'], yhat_multipoly))
# #Toma como argumentos el valor actual y el predicho de la variable objetivo


# #CALCUO DEL R2

# # fit the model 
# # Find the R^2
print('The R2 entre el modelo DE MULTIPLE LINEAL REGRESSION CON PIPELINES:: ',pipe.score(Z, df['price']))

# #Analysis:We can say that ~84.633 % of the variation 
# of price is explained by this multiple polynomial regression "multi_fit".


#CALCULO DE MSE Y R2 MODELO POLYNOMIAL MULTIPLE

prn=PolynomialFeatures(degree=2)

Zt=prn.fit_transform(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

#creo el objeto de regresion lineal para efectuar la predicción de los datos polinomicos
lrn=LinearRegression()

lrn.fit(Zt,df['price'])

yhatn=lrn.predict(Zt)
print ('Coefficients MPR: ', lrn.coef_)
print ('Intercept MPR: ',lrn.intercept_)

print('The mean square CON MPR metodo normal: ', \
      mean_squared_error(df['price'], yhatn))
# #Toma como argumentos el valor actual y el predicho de la variable objetivo


# #CALCUO DEL R2

# # Find the R^2
print('The R2 entre el modelo de REGRESION POLINOMIAL MULTIPLE NORMAL:: ',lrn.score(Zt, df['price']))

#IMPORTANTE:
# 
# ME DA IGUAL EL R2 ENTRE EL METODO NORMAL DE MPR Y USANDO PIPELINES
#PIPELINES APROXIMA EL MEJOR ORDEN PARA EL POLINOMIO




#***************************************+PREDICTION AND DECISITION MAKING







#EJEMPLO USANDO HIGHWAY-MPG AND PRICE SIMPLE LINEAR REGRESSION MODEL (SLR)


#Lets predict the price of a car with 30 highway.pmg

#si highway-mpg sube un galon el precio disminuye 821.73

#en ocasiones el modelo puede dar valores que no parecen razonables como un precio negativo
#puede ser que el modelo lineal no sea el adecuado o que no se tenian datos en ese rango para armar el modelo

# por ejemplo en el rango de 1 a 100 para highway-mpg aparecen valores negativos
# En este caso aunque se presenten estos valores negativos, 
# es poco probable que un automóvil tenga un consumo de combustible en ese rango,
# por lo que nuestro modelo pueda ser valido


#SIN EMBARGO EN ESTE CASO EN PARTICULAR SE SABE QUE EL MODELO LINEAL NO ERA EL ADECUADO
#PARA ESTA VARIABLE

#Genero la secuencia de 1 a 100
new_input=np.arange(1, 100, 1).reshape(-1, 1)

#Fit the model
lm.fit(X, Y)

#Produce a prediction:
ypredictSLR = lm.predict(new_input)
X_predict=[[30]] #si quiero poner dos valores [[0],[30]]

print('Prueba del Modelo SLR highway-mpg and price using values  in highway to (1-100):',ypredictSLR)
print('Prueba con un solo valor (30) en highway.mpg SLR',lm.predict(X_predict))
#NOTA:EL ARGUMENTO DE PREDICT PIDE UN ARREGLO EN 2D

#muestro los primeros4 valores obtenidos

#We can plot the data:
plt.title('Prueba del Modelo SLR highway-mpg and price using values  in highway to (1-100)')
plt.xlabel('highway-mpg')
plt.ylabel('Predicted price')
plt.plot(new_input, ypredictSLR)
plt.show()


#EJEMPLO USANDO MLR

lmm.fit(Z,df['price'])
X_predictm=[[115,2900,159,34]]
print(df.loc[0:4,['horsepower','curb-weight','engine-size','highway-mpg']])
print('Prueba con un solo valor en el MLR',lmm.predict(X_predictm))

#EJEMPLO SIMPLE POLYNOMIAL FIT CON PIPELINES

#EN SIMPLE POLYNOMIAL FIT RECORDAR QUE EL METODO ME DA LA ECUACION DEL POLINIOMIO PARA REEMPLAZAR
#PERO TAMBIEN LO PUEDO HACER CON PIPELINES, ME DA UN RESULTADO CON UNA PEQUEÑA DIFERENCIA, TAMBIEN SE PUEDE
#HACER UN SIMPLE POLYNOMIAL FIT CON EL METODO FIT_TRANSFORM Y LUEGO UNA LR PARA PREDECIR

pipe1=Pipeline(Input)
pipe1.fit(X,Y)
ypipeslre=pipe1.predict(X_predict)
print("Predicción USANDO PIPELINES CON NORMALIZACION, TRANSFORMACION Y LR (SIMPLE POLINOMIAL FIT)",'\n',ypipeslre[0:4])
#NOTA: ME DA EL MISMO RESULTADO USANDO EL SISTEMA COMPLETO CON PIPELINES QUE EL PROCESO NORMAL DEL SLR


#EJEMPLO MULTIPLE POLYNOMIAL FIT CON PIPELINES

pipe2=Pipeline(Input)
pipe2.fit(Z,df['price'])
Z = Z.astype(float)
yhatpipemp=pipe2.predict(X_predictm)
print("Predicción MULTIPLE POLINOMIAL FIT usando pipelines normalizacion,transformacion,regression",'\n',yhatpipemp[0:4])






#******************************************CONCLUSIONS EVALUATING THE MODELs*******************************


# Decision Making: Determining a Good Model Fit
# Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?

# What is a good R-squared value?
# When comparing models, the model with the higher R-squared value is a better fit for the data.

# What is a good MSE?
# When comparing models, the model with the smallest MSE value is a better fit for the data.

# Let's take a look at the values for the different models.
# Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.

# R-squared: 0.49659118843391759
# MSE: 3.16 x10^7
# Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.

# R-squared: 0.80896354913783497
# MSE: 1.2 x10^7
# Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.

# R-squared: 0.6741946663906514
# MSE: 2.05 x 10^7





# Simple Linear Regression Model (SLR) vs Multiple Linear Regression Model (MLR)



# Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and even act as noise. As a result, you should always check the MSE and R^2.

# In order to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.

# MSE: The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.
# R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809).
# This R-squared in combination with the MSE show that MLR seems like the better model fit in this case compared to SLR.

# This R-squared in combination with the MSE show that MLR seems like the better model fit in this case compared to SLR.
#MLR WINS


# Simple Linear Model (SLR) vs. Polynomial Fit



# MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.
# R-squared: The R-squared for the Polynomial Fit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.
# Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting "price" with "highway-mpg" as a predictor variable.
#POLYNOMIAL FIT WINS



# Multiple Linear Regression (MLR) vs. Polynomial Fit


# MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
# R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.
#MLR WINS.

# Multiple Linear Regression (MLR) vs.  MULTIPLE Polynomial Fit MPR
#EL R2 de MPR es mayor que el de SLR, ademas el MSE en MPR es menor que SLR
#Hasta este punto gana este modelo, sin embargo  MAS ADELANTE SE DEBE MIRAR DE NUEVO ESTOS
#VALORES CON EL MODELO PORBANDOLO CON DATOS NUEVOS O DATOS DE PRUEBA Y EN BASE A ELLOS DECIDIR


# Conclusion


# Comparing these three models, we conclude that the MLR model 
#CAMPRING THE FOUR MODELS THE BEST IS MPR
# is the best model to be able to predict price from our dataset. 
# This result makes sense since we have 27 variables in total and we 
# know that more than one of those variables are potential predictors of the final car price.











#***************************4. MODEL EVALUATION AND REFINEMENT/ EVALUACIÓN DEL MODELO Y REFINAMIENTO***********************


#***************TRAINING DATA SETS



# DIVIDA SUS DATOS

# UTILICE LOS DATOS DE MUESTRA O LOS DATOS DE ENTRENAMIENTO PARA ENTRENAR EL MODELO

# El resto de los datos, llamados datos de prueba, se utilizan como datos fuera de la muestra.
# Estos datos se utilizan luego para aproximar el rendimiento del modelo en el mundo real.

#Ej:70% de los datos para entrenar los datos, y el otro 30% para datos de prueba
#Pero cuando se termine la prueba se deben usar todos los datos para entrenar el modelo


# An important step in testing your model is to split your data into training and 
# testing data. We will place the target data price in a separate dataframe y_data:

y_data = df['price']

#Para los datos de x, tomo todos los datos del df eliminando la columna price
#en realidad deberian ser las variables predictoras que se hallaron anteriomente,
#pero en este caso se hará con todas los features/columnas/ o variables del dataframe
x_data=df.drop('price',axis=1)



#Now, we randomly split our data into training and testing data using the function train_test_split.

#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=1)
#Me arroja las variables x and y como arreglos para training y testing, test_size: es el porcentaje de datos 
# para testing

# x_data: features or independent variables
# y_data: dataset target: dfffprice']
# x_train, y_train: parts of available data as training set
# _test, y_test: parts of available data as testing set
# test_size: percentage of the data for testing (here 30%)
#random_state: number generator used for random sampling

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#We create a Linear Regression object:
lre=LinearRegression()

#We fit the model using the feature "horsepower":
lre.fit(x_train[['horsepower']], y_train)

#Let's calculate the R^2 on the test data:
print('R^2 de horsepower using split with test data SLR',lre.score(x_test[['horsepower']], y_test))
#En modelos de regresion lineal se usan los x train y y_train que arroja el split test
#en el test polinomico se debe poner el x_train y el x_test transformado

#We can see the R^2 is much smaller using the test data compared to the training data.
print('R^2 de horsepower using split with training data SLR',lre.score(x_train[['horsepower']], y_train))



#********GENERALIZATION ERROR




#El error de generalización es una medida de qué tan bien funcionan nuestros datos en
#predecir datos nunca antes vistos

#El error que obtenemos usando nuestros datos de prueba es una aproximación al generalization error



#¿EN QUE AFECTA EL GENERALIZATION ERROR?

#Cuando grafico un MLR usando los datos training me da mas ajustado, si uso
#los datos de test me da un grafico menos ajustado y que se ajusta mas al mundo real.
#esta diferencia es debida al genralization error


#LOW  TESTING DATA QUE PASA CON LOS RESULTADOS Y EL GENERALIZATION ERROR?

#Si uso un gran porcentaje para el training y menos para el testing ej 90% y 10% obtengo un resultado
#mas cercano al generalization error entrenando varios tipos de muestra con este resultado,
#SIN EMBARGO, NO HAY PRECISION, CADA RESULTADO QUE OBTENGO CON LAS DIFERENTES MUESTRAS SON MUY DIFERENTES
#ENTRE SI.


#HIGHER TESTING DATA QUE PASA CON LOS RESULTADOS Y EL GENERALIZATION ERROR?

#PERO SI USO UN MAYOR PORCENTAJE PARA EL TESTING OBTENGO RESULTADOS MAS PRECISOS, NO TAN
#CERCANOS AL GENeRALIZATION ERROR (EXACTITUD), PERO SUS RESULTADOS COINCIDEN CADA VEZ QUE PRUEBO CON DIFERENTES MUESTRAS.


#¿COMO LOGRO ACERCAR UN MODELO CON UN TESTING DATA MAS ALTO AL GENERALIZATION ERROR?
#para solucionar este problema se usa el CROSS VALIDATION





#********************CROSS VALIDATION*****************



#¿COMO LOGRO ACERCAR UN MODELO CON UN TESTING DATA MAS ALTO AL GENERALIZATION ERROR?
#para solucionar este problema se usa el CROSS VALIDATION

#Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation. 
# Let's go over several methods that you can use for cross-validation.

#EN CROSS VALIDATIONS SE DIVIDEN LOS DATOS EN GRUPOS IGUALES, EL METODO REALIZA DIFERENTES PRUEBAS
# DE TAL FORMA  QUE CADA FRUPO  DE DATOS SE USE TANTO EN TEST COMO EN TRAINING 
# Y AL FINAL SE USAN LOS RESULTADOS PROMEDIO COMO UNA ESTIMACIÓN DEL ERROR FUERA DE MUESTRA (OUT-OF-SAMPLE-ERROR)

#from sklearn.model_selection import cross_val_score

#PARA LA VALIDACIÓN CRUZADA PUEDO INTRODUCIR LOS SIGUIENTES PARAMETROS PARA UN MODELO POLINOMIAL

#LRE: PARA MODELO POLINOMIAL IGUAL TENGO QUE UTILIZAR UN OBJETO DE REGRESSION LINEAL LUEGO DE TRANSFORMAR LOS DATOS
#EN VEZ DE X_DTA UTILIZO UN X_DATA TRANSFORMADO ASI COMO ELY_DATA



#AQUI EL CROSS VALIDATION ES PARA UN SLR


#We input the object, the feature ("horsepower"), and the target data (y_data). 
# The parameter 'cv' determines the number of folds. In this case, it is 4.
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

#The default scoring is R^2. Each element in the array has the average R^2 value for the fold:
print("R2 obtenidos en cada validación cruzada  de horsepower (cross validation): ",Rcross)
#ESTOS RESULTADOS SE USAN COMO UNA ESTIMACIÓN DEL ERROR FUERA DE MUESTRA (OUT-OF-SAMPLE-ERROR)
#POR DEFECTO ME ARROJA LOS R2 PUEDO PUEDO DARLE OTRAS METRICAS

#We can calculate the average and standard deviation of our estimate:
print("The mean of R2 folds of horsepower using cross validation are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
#ESTOS RESULTADOS SE USAN COMO UNA ESTIMACIÓN DEL ERROR FUERA DE MUESTRA (OUT-OF-SAMPLE-ERROR)

#Analysis: De esta forma al usar una validación cruzada se usa un buen porcentaje de datos para el test
#obteniendo resultados precisios en cada intento y obteniiendo un R2 mas alto, lo cual indica
#que se esta mas cerca del generalization error


#USANDO OTRA METRICA PARA EVALUAR EL MODELO CON HORSEPOWER USANDO CROSS VALIDATION
#We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.
nmse_cross=-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

print("Error cuadratico negativo modelo horsepower usando cross validation:",nmse_cross)
print("The mean of neg_mean_squared_error folds of horsepower using cross validation are", nmse_cross.mean(), "and the standard deviation is" , nmse_cross.std())


#****CROSS VALIDATION PREDICT

# You can also use the function 'cross_val_predict' to predict the output. 
# The function splits up the data into the specified number of folds, with one fold for testing 
# and the other folds are used for training. First, import the function:

#from sklearn.model_selection import cross_val_predict

#En este momento se esta haciendo con horsepower, sin embargo se puede introducir cualquier objeto
#que tenga un modelo de regresion, en este caso un pipe2
yhat = cross_val_predict(pipe2,Z, y_data,cv=4)
print('Predicccion usando validación cruzada (cross validation)',yhat[0:5])





#************************************Overfitting, Underfitting and Model Selection



#Underfitting:Ocurre cuando asigno una regresion lineal o un polinomio de bajo orden que no se ajusta
#a los datos de manera correcta

#Overfftting:Ocurre cuando asigno un polinomio de grado mayor al necesario,
#Ej: si tengo un polinomio de grado 8 que realiza un buen ajuste a los valores actuales y asigno un polinomio 
# de grado mas alto este puede tender a tener un gran error y un oscilamiento en el ajuste

#Recordar que elmodelo esta dado por y(x)+noise, el oise esta presente en todos los modelos


"""
   Error Mse y

          Under fit            Over fit
        I|   x   |       |    x
        I|o    x |    x  |
        I|   o   |  x----|-----------------Este intervalo representa el error del polinomio adecuado, 
        I|     o | o  o  | o o o----------- este se sigue presentando por el noise, el polinomio no es el mejor
        Illllllllll!lllllllll!lllllllllll
                Order 8    Order 11          
        Order x

        o:Training error:Disminuye el Error Mse a medida que aumento el orden del polinomio
        x:Test Error: Disminuye el Error Mse hasta que se logra el orden correcto del polinomio para ajustar el modelo
        pero aumenta a medida que aumento el arden del polinomio alejandolo de su ajuste correcto.

        Se debe tomar como referencia el Test Error ya que me indica cual es el desempeño del modelo
        en el mundo real, y este error puede aumentar si ajusta erroneamente a un polinomio de orden mayor

        Ademas el R2 disminuye si aumentamos erroneamente el orden polinomio,
        puedo validar esto graficando R2 vs orden de cada polinomio ajustado al modelo
"""


#********MLR TRAIN AND TEST DATA PLOT


#Let's create Multiple Linear Regression objects and train the model using 
# 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.

lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
#Recordar que x_train y y_train salieron de aplicar una funcion al price y a los demas datos del df
#tambien puedo poner x_train[z]

yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_testmlr=x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
print('R^2 DATOS DE PRUEBA USANDO MLR',lr.score(x_testmlr, y_test))


#DEFINO LA FUNCION PARA GRAFICAR LA DISTRIBUCIÓN PARA COMPARAR EL TRAIN Y EL TEST
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


#DEFINO LA FUNCION PARA GRAFICAR LOS POLINOMIOS O LR DE LOS MODELOS TRAIN Y TEST
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()


#Let's examine the distribution of the predicted values of the training data.
Title = 'Distribution  Plot MLR of  Predicted Value Using Training Data vs Training Data Prediction Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

#Analysis:So far, the model seems to be doing well in learning from the training dataset. 
# But what happens when the model encounters new data from the testing dataset? When the 
# model generates new values from the test data, we see the distribution of the predicted values
#  is much different from the actual target values.

Title='Distribution  Plot MLR  of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

#Analysis:Comparing Figure 1 and Figure 2, it is evident that the distribution of the test data
#  in Figure 1 is much better at fitting the data. This difference in Figure 2 is apparent in the
#  range of 5000 to 15,000. This is where the shape of the distribution is extremely different. 
# Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.



#*****POLYNOMIAL REGRESSSION TEST AND TRAINING PLOT


#Overfitting

# El sobreajuste ocurre cuando el modelo se ajusta al ruido, pero no al proceso subyacente. 
# Por lo tanto, al probar su modelo con el conjunto de prueba, su modelo no funciona tan bien ya 
# que está modelando ruido, no el proceso subyacente que generó la relación. 
# Vamos a crear un modelo polinomial de grado 5.

# Usemos el 55 por ciento de los datos para entrenamiento y el resto para pruebas:

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

#Now, let's create a Linear Regression model "poly" and train it.
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
#Hace una regresion lineal con los datos transformados a orden 5, en si es una regresion
#polinomial de orden 5

#We can see the output of our model using the method "predict." We assign the values to "yhat".
yhat = poly.predict(x_test_pr)
print("prediccion usando polinmio con el metodo normal", yhat[0:4])
print("Predicción MULTIPLE POLINOMIAL FIT usando pipelines normalizacion,transformacion,regression",'\n',yhatpipe[0:4])



#Let's take the first five predicted values and compare it to the actual targets.
print("Predicted values of price with x_test using simple polynomial regression 5 order:", yhat[0:4])
print("True values with y_test (valores actuales del precio de prueba):", y_test[0:4].values)

# We will use the function "PollyPlot" that we defined at the beginning of the 
# lab to display the training data, testing data, and the predicted function.
#PollyPlot(x_train[['horsepower','curb-weight','engine-size','highway-mpg']], x_test[['horsepower','curb-weight','engine-size','highway-mpg']], y_train, y_test, poly,pr)

# x_train=x_train.to_numpy()
# x_test=x_test.to_numpy()
PollyPlot(x_train.loc[:,'horsepower'], x_test.loc[:,'horsepower'], y_train, y_test, poly,pr)

#Analysis:Figure 3: A polynomial regression model where red dots represent training data, green dots represent test data, 
# and the blue line represents the model prediction.
#We see that the estimated function appears to track the data but around 200 horsepower, 
# the function begins to diverge from the data points.

#R^2 of the training data:
poly.score(x_train_pr, y_train)
print('R^2 de horsepower using split with training Polynomial Regression training',poly.score(x_train_pr, y_train))

#R^2 of the test data:
poly.score(x_test_pr, y_test)
print('R^2 de horsepower using split with training data Polynomial Regression testing',poly.score(x_test_pr, y_test))


#COMO CAMBIA EL R2 SEGUN EL ORDEN DEL POLINOMIO?

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))
    print('R2 horsepower in order',n,'es:',lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data for SPR')
plt.text(3, 0.75, 'Maximum R^2 ')
#Al analizar el grafico pongo esta marca es 3 y o.75 que es donde aporximadamente
#mejor se ajusta el R2
plt.show()   

#Analysis:We see the R^2 gradually increases until an order three polynomial is used. 
# Then, the R^2 dramatically decreases at an order four polynomial.



#***** MULTIPLE POLYNOMIAL REGRESSSION TEST AND TRAINING PLOT




#creo el objeto para transformar los datos polinomicos
pr1=PolynomialFeatures(degree=2)

x_train_pr1=pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

x_test_pr1=pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

print('features de los datos transformados para la regresion multiple polinomica',x_train_pr1.shape)

#creo el objeto de regresion lineal para efectuar la predicción de los datos polinomicos
poly1=LinearRegression()

poly1.fit(x_train_pr1, y_train)

yhat_test1=poly1.predict(x_test_pr1)


#Grafico los datos del test data
Title='Distribution  Plot of  Predicted Value Using MPR Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)
print('R2 MPR using test data in order 2',poly1.score(x_test_pr1, y_test))


y_test1 = np.asanyarray(y_test)
print('R2 USING Y_TEST AND YHAT MPR in order 2',r2_score(y_test1, yhat_test1))

#ANALYISIS:
#AMBOS R2 SON IGUALES TANTO HACIENDOLO CON X_TEST_PR1 Y Y_TEST ASI COMO CON Y_TEST Y YHAT, 
# TODOS LOS R2 EN ESTE DOCUMENTO ESTAN CORRECTAMENTE CALCULADOS



#PollyPlot(x_train[['horsepower','curb-weight','engine-size','highway-mpg']].to_numpy(), x_test[['horsepower','curb-weight','engine-size','highway-mpg']].to_numpy(), y_train, y_test, poly1,pr1)
#NO FUNCIONA YA QUE LOS MODELOS MULTIPLES NECESITAN SER GRAFICADOS EN UN HIPERPLANO O EN SU DEFECTO EN UNA FUNCION DE DISTRIBUCIÓN

#Analysis: The predicted value is higher than actual value for cars where the price $10,000 range, 
# onversely the predicted price is lower than the price cost in the $30,000 to $40,000 range. 
# As such the model is not as accurate in these ranges.


#COMO CAMBIA EL R2 SEGUN EL ORDEN DEL POLINOMIO?

Rsqu_test1 = []

order1 = [1, 2, 3, 4]
for n in order1:
    pr1 = PolynomialFeatures(degree=n)
    
    x_train_pr1 = pr1.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
    
    x_test_pr1 = pr1.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])    
    
    lr.fit(x_train_pr1, y_train)
    
    Rsqu_test1.append(lr.score(x_test_pr1, y_test))
    print('R2 MPR in order using test data',n,'es:',lr.score(x_test_pr1, y_test))



plt.plot(order1, Rsqu_test1)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data for MPR')
plt.text(1, 0.78, 'Maximum R^2 ')
#Al analizar el grafico pongo esta marca es 3 y o.75 que es donde aporximadamente
#mejor se ajusta el R2
plt.show()   


#Analysis: AL EXAMINAR LOS R2 DEL MODELO DE REGRESSION POLYNOMIAL MULTIPLE SE OBSERVA
#QUE CON DATOS DE PRUEBA EL MODELO DE REGRESION POLYNOMIAL TIENE UN MEJOR DESEMPEÑO
# QUE EL MLR, YA QUE AUNQUE EL QUE MEJOR SE AJUSTE ES EL ORDEN 1, AL HACER LA 
#TRANSFORMACIÓN DE LOS DATOS SE AJUSTA MEJOR LA PREDICCIÓN, POR SUPUESTO SI SE SUBE EL
# ORDEN DEL POLINOMIO EVIDENCIAMOS UN OVERFITTING






#******************************++RIDGE REGRESSION






#CONTROLA LA MAGINITUD DE LOS COEFICIENTES DE LOS POLINOMIO INTRODUCIENDO EL PARAMETRO ALPHA

#ALPHA IS SELECTED BEFORE FITTING OR TRAINNIG THE MODEL
#SI INCREMENTO EL VAÑLOR DE ALPHALOS COEFICIENTES  DEL POLINOMIO VAN DISMINUYENDO
#UN VALOR DE ALPHA DEMASIADO ALTO PODRIA CAUSAR UN UNDERFITTING YA QUE EL COEFICIENTE SE VOLVERIA 0 O CERCANO A 0


#AL HACER EL PROCESO SE SELECCIONA EL ALPHA QUE ME ARROJE UN MEJOR VALOS DE R2 O DE ALGUNA OTRA MEDIDA COMO EL MSE

#In this section, we will review Ridge Regression and see how the parameter alpha changes the model. Just a note
# , here our test data will be used as validation data.



#Let's perform a degree two polynomial transformation on our data.

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

#Let's import Ridge from the module linear models.
#from sklearn.linear_model import Ridge

#Let's create a Ridge regression object, setting the regularization parameter (alpha) to 0.1
RigeModel=Ridge(alpha=1)

#Like regular regression, you can fit the model using the method fit.
RigeModel.fit(x_train_pr, y_train)

#Similarly, you can obtain a prediction:
yhat = RigeModel.predict(x_test_pr)

#Let's compare the first five predicted samples to our test set:
print('predicted MPR USING RIDGE REGRESSION:', yhat[0:4])
print('test set y_test :', y_test[0:4].values)


# We select the value of alpha that minimizes the test error. 
# To do so, we can use a for loop. We have also created a progress bar to see how many 
# iterations we have completed so far.

#from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

print('MPR 2 order using Ridge Regression with alpha:',alpha,'gives a validation (testing) R2:',test_score)
print('MPR 2 order using Ridge Regression with alpha:',alpha,'gives a training R2:',train_score)

#SI AUMENTA EL ALFA EL R2 EN VALIDATION AUMENTA (SI AUMENTA DEMASIADO EL ALPHA PUEDE OCURRIR UNDERFITTING)
#SI AUMENTA EL ALFA EL R2 EN TRAINING DISMINUYE

#We can plot out the value of R^2 for different alphas:
width = 12
height = 10
plt.figure(figsize=(width, height))
plt.title(" Cambio del R^2 MPR Ridge Regression Validation and training data")
plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

#Analysis:
# The blue line represents the R^2 of the validation data, and the red line represents the R^2 
# of the training data. The x-axis represents the different values of Alpha.

# Here the model is built and tested on the same data, so the training and test data are the same.

# The red line in Figure 4 represents the R^2 of the training data. As alpha increases the R^2 decreases. 
# Therefore, as alpha increases, the model performs worse on the training data

# The blue line represents the R^2 on the validation data. As the value for alpha increases, 
# the R^2 increases and converges at a point.





#**************GRID SEARCH




#GRID SEARCH PERMITE ESCANEAR MULTIPLES PARAMETROS LIBRES (HYPERPARAMETROS) CON POCAS LINEAS DE CODIGO

#IMPORTANTE:PARAMETROS COMO ALFA NO SON PARTE DEL FITTING O DEL TRAINING DEL MODELO
#SON HYPERPARAMETERS QUE AYUDAN A MODFICAR EL MODELO PARA EVITAR EL OVERFITTING O EL UNDERFITTING


#UNA VEZ SE INTRODUCEN LOS PARAMETROS EN EL GRID SERCG OBTENGO VARIOS MODELOS Y SELECCIONO
# EL QUE MENOR ERROR TENGA O MAYOR R2

#PARA EL GRID SEARCH SE DIVIDE EL PROCESO EN 3 PARTES

#TRAINING      VALIDATION   TEST (SE PRUEBAN LOS HYPERPARAMETROS EN CADA TIPO DE DATOS Y SE OBTIENE EL MEJOR MODELO)




# The term alpha is a hyperparameter. Sklearn has the class GridSearchCV to make 
# the process of finding the best hyperparameter simpler.
# Let's import GridSearchCV from the module model_selection.

#from sklearn.model_selection import GridSearchCV

#We create a dictionary of parameter values:
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]},{'normalize':[True,False]}]
#el parametro normalize me dice si el paametro esta normalizado o no

#Create a Ridge regression object:
RR=Ridge()

#Create a ridge grid search object:
Grid1 = GridSearchCV(RR, parameters1,cv=4)

#In order to avoid a deprecation warning due to the iid parameter, we set the value of iid to "None".

#Fit the model:
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

#The object finds the best parameter values on the validation data. We can obtain the estimator
# with the best parameters and assign it to the variable BestRR as follows:
BestRR=Grid1.best_estimator_
print("Ridge Search best estimator for the model: ",\
    BestRR)

#We now test our model on the test data showing the R2:
BestR2_RidgeReg=BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
print("R2 DEL MEJOR MODELO ULIZANDO REGRESSION MULTIPLE CON RIDGE SEARCH:", BestR2_RidgeReg)

#The resulting scores of the different free parameters are stored in this dictionary:
scores=Grid1.cv_results_
print("Resultados obtenidos de los paametros ingresados en el grid search",\
    scores)




#CONCLUSION DEFINITIVA

#USAR UN MPR DE ORDEN UNO 
#EL MODELO DEFINITIVO DE DEBE REALIZAR CON TODO LOS DATOS DISPONILBLES
#CON TODOS LOS DATOS DISPONIBLES ME DA UN R2 MAS GRANDE UN POLINOMIO DE MAYOR ORDEN? SI

#PERO AL MOMENTO DE PROBARLO CON NUEVOS DATOS SE PRODUCE UN OVERTTIFITNG

#ASI QUE LA CONCLUSION FINAL ES UN MPR DE ORDEN UNO CON TODOS LOS DATOS



#COMO CAMBIA EL R2 SEGUN EL ORDEN DEL POLINOMIO?

Rsqu_testp = []

orderp = [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14]
for n in orderp:
    pr5 = PolynomialFeatures(degree=n)
    
    Z_tr=pr5.fit_transform(Z)
    lr5=LinearRegression()
    lr5.fit(Z_tr,df['price'])
    
    Rsqu_testp.append(lr5.score(Z_tr,df[['price']]))
    print('R2 MPR with original data in order',n,'es:',lr5.score(Z_tr,df[['price']]))

plt.plot(orderp, Rsqu_testp)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using MPR with original data')
plt.text(3, 0.75, 'Maximum R^2 ')
#Al analizar el grafico pongo esta marca es 3 y o.75 que es donde aporximadamente
#mejor se ajusta el R2
plt.show()   














































































































