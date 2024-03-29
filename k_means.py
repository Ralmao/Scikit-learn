import pandas as pd

from sklearn.cluster import MiniBatchKMeans


if __name__ =="__main__":
    
    path = 'data/candy_a74a49fd-6364-4c16-9381-406cdb66f338.csv'
    dataset = pd.read_csv(path)

    print(dataset.head(5))

    x = dataset.drop('competitorname', axis=1)
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)

    print('')
    print('Total de centros:', len(kmeans.cluster_centers_))

    print('')
    print('Predicciones:', kmeans.predict(x))

    dataset['Grupo'] = kmeans.predict(x)
    print(dataset)

    # Ahora mando los datos a un archivo excel :)

    #writer = pd.ExcelWriter('data/candy_a74a49fd-6364-4c16-9381-406cdb66f338.csv', engine='xlsxwriter')
    #dataset.to_excel(writer, sheet_name='usuario')
    #writer.save()
    