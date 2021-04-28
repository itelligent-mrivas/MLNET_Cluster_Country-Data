# MLNET_Cluster_Country-Data
Construcción de un Modelo de Clustering a partir de ML.NET para agrupar 167 países según sus datos Socioeconómicos.  
Para la construcción del mencionado Modelo, en primer lugar, se crea la clase de representación de observaciones a través de una tabla de datos, (extraída de Kaggle), la cual es transformada en un IDataView.  
Posteriormente se establecen dos pipeline, los cuales contienen, por un lado, las transformaciones de los datos y por el otro el Algoritmo de Entrenamiento de ML.NET seleccionado, KMeans, y se entrena el Modelo.  
Se concluye el programa evaluando el Modelo a través de métricas características de los Modelos de Clustering y visualizando los clústers designados a cada país a través de dos formatos diferentes, (forma gráfica y una tabla). 
