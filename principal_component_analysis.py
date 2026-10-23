import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

spec_power_data_file = 'Data/winequality-red.xlsx'
reader = pd.read_excel(spec_power_data_file, header=0)
columns = reader.columns
data_csv=reader.to_numpy()
#Imprimimos la forma(dimensiones) de nuestra matriz
print("Dimensiones de la matriz de datos original",data_csv.shape)
index=['Sample '+str(ind) for ind in range(1,len(data_csv)+1)]
data=pd.DataFrame(data=data_csv,columns=columns,index=index)
data=data.drop(['quality'],axis=1)
#Datos del nuevo DataFrame
print(data.tail(len(data)))
#como nos damos cuenta estamos pasando la transpuesta,
#dicha funcion scale lo que hace es centrar y escalar los datos
scaled_data=preprocessing.scale(data)
#print(data.shape)

pca=PCA()
#Aqui se hacen los calculos de PCA
pca.fit(scaled_data)
#Aqui generamos las coordenadas para una grafica de PCA basado en los datos escalados
pca_data=pca.transform(scaled_data)
#calculamos el porcentaje de variacion de cada componente principal
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
#creamos los labels, uno por cada compoenente principal
labels=['PC'+str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Porcentaje de varianza explicada')
plt.xlabel('Componente principal')
plt.title('Screen plot')
plt.show()
#imprimiendo el porcentaje de varianza explicada por cada componente
for x in range(len(per_var)):
    print("PC"+str(x+1), "Percentage", per_var[x])

