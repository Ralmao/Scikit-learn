import pandas as pd 

from sklearn.cluster import MeanShift 

if __name__ == "__main__":

   # Cargamos el dataset 
    dataset = pd.read_csv('data/candy_a74a49fd-6364-4c16-9381-406cdb66f338.csv') 
    #print(dataset.head(5))  

    # Vamos a eliminar la columna
    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    # Aquí el algoritmo nos devolvio 3 clusters, porque le pareció que esa era la cantidad 
    # correcta teniendo en cuenta como se distrubuye la densidad de nuestros datos 
    print(max(meanshift.labels_))
    print("="*64)
    # Imprimamos la ubicación de los centros que puso sobre nuestros datos. Hay que recordar que estos algoritmos 
    # crean un centro y a partir de ahí se ajuztan a todos los datos que lo rodean  
    print(meanshift.cluster_centers_)
    
    # Los arreglos lo integramos a nuestros datasets 
    dataset['meanshift'] = meanshift.labels_  
    print("="*64)
    print(dataset)  
