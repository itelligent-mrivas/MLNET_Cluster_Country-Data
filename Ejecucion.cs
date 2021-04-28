using System;

using System.IO;

using System.Collections.Generic;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

using OxyPlot;
using OxyPlot.Series;

namespace MLNET_Cluster_Country_Data
{
    class Ejecucion
    {
        //Rutas de acceso de dataset y Modelos
        static readonly string _DataPath = @"./DATOS/Countrydata.csv";
        static readonly string _salida_trainDataPath = @"./DATOS/trainData.csv"; 
        static readonly string _salida_transformationData = @"./DATOS/transformationData.csv";
        static readonly string _salida_plotDataPath = @"./DATOS/plot_cluster.svg";
        static readonly string _salida_ResultDataPath = @"./DATOS/ClusteringResult.csv";
        static readonly string _salida_modelPath = @"./DATOS/Model.zip";
        static void Main(string[] args)
        {

            //###############################################################
            //INICIALIZACIÓN DEL PROCESO
            //###############################################################

            //Inicialización de mlContext; utilización del seed para replicidad
            MLContext mlContext = new MLContext(seed: 1);

            //Definición de las clases de los datos de entrada: 
            //  -Clase Observaciones: CountryObservation

            //Carga de datos
            IDataView originalFullData = mlContext.Data
                .LoadFromTextFile<CountryObservation>(
                _DataPath,
                separatorChar: ',',
                hasHeader: true);


            //###############################################################
            //CONSTRUYE EL CONJUNTO DE DATOS (DATASET)
            //###############################################################

            IDataView trainingDataView = originalFullData;

            //Guardamos dataset trainingDataView
            using (var fileStream = File.Create(_salida_trainDataPath))
            {
                mlContext.Data.SaveAsText(trainingDataView, fileStream, separatorChar: ';', headerRow: true,
                    schema: true);
            }


            //###############################################################
            //SELECCIÓN DE VARIABLES
            //###############################################################

            //Suprimimos del esquema IDataView lo que no seleccionemos como features
            string[] featureColumnNames = trainingDataView.Schema.AsQueryable()
                .Select(column => column.Name)
                .Where(name => name != "country")//no aporta información
                .ToArray();

            //###############################################################
            //TRANFORMACIÓN DE LOS DATOS DEL MODELO --> pipeline
            //###############################################################

            //Concatena
            IEstimator<ITransformer> pipeline = mlContext.Transforms.Concatenate("Features",
                featureColumnNames)
            //Normalizado de las Features
            .Append(mlContext.Transforms.NormalizeMinMax(inputColumnName: "Features",
               outputColumnName: "FeaturesNormalized"));

            //Guardamos dataset transformedData        
            IDataView transformedData =
                pipeline.Fit(trainingDataView).Transform(trainingDataView);
            using (var fileStream = File.Create(_salida_transformationData))
            {
                mlContext.Data.SaveAsText(transformedData, fileStream, separatorChar: ';', headerRow: true,
                    schema: true);
            }


            //###############################################################
            //SELECCIÓN DEL ALGORITMO DE ENTRENAMIENTO --> trainingPipeline
            //###############################################################

            //***************************************************************
            //1. K-Means
            //***************************************************************            

            //Selección del Número de Clusters
            int k = 4; 

            //Opciones K-Means
            var options = new KMeansTrainer.Options
            {
                FeatureColumnName = "FeaturesNormalized",
                NumberOfClusters = k,
                MaximumNumberOfIterations = 5800,
                OptimizationTolerance = 1e-6f
            };

            //K-Means
            var trainer_km = mlContext.Clustering.Trainers.KMeans(options);

            //Se añade el Algoritmo al pipeline de transformación de datos
            IEstimator<ITransformer> trainingPipeline_km = pipeline.Append(trainer_km);


            //###############################################################
            //ENTRENAMIENTO DEL MODELO
            //###############################################################

            Console.WriteLine($"\n**************************************************************");
            Console.WriteLine($"* Entrenamiento del Modelo calculado con el Algoritmo K-Means   ");
            Console.WriteLine($"*-------------------------------------------------------------");
            var watch_km = System.Diagnostics.Stopwatch.StartNew();
            var model_km = trainingPipeline_km.Fit(trainingDataView);
            watch_km.Stop();
            var elapseds_km = watch_km.ElapsedMilliseconds * 0.001;
            Console.WriteLine($"El entrenamiento K-Means ha tardado: {elapseds_km:#.##} s\n");


            //###############################################################
            //EVALUACIÓN DEL MODELO
            //###############################################################

            //Transformación del IDataView trainingDataView
            var predictions_km = model_km.Transform(trainingDataView);

            //Calculo de las métricas de cada Modelo
            var metrics_km = mlContext.Clustering.Evaluate(predictions_km, 
                scoreColumnName: "Score", 
                featureColumnName: "FeaturesNormalized");

            //Mostramos las métricas K-Means
            Console.WriteLine($"\n**************************************************************");
            Console.WriteLine($"* Métricas para el Modelo calculado con el Algoritmo K-Means      ");
            Console.WriteLine($"*-------------------------------------------------------------");
            Console.WriteLine($"*       K-Means Average Distance:  {metrics_km.AverageDistance:#.##}");
            Console.WriteLine($"*       K-Means Davies Bouldin Index:  {metrics_km.DaviesBouldinIndex:#.##}");
            Console.WriteLine($"*       K-Means Normalized Mutual Information:  {metrics_km.NormalizedMutualInformation:#.##}");


            //###############################################################
            //SELECCIÓN MODELO
            //###############################################################            

            //Guardamos el Modelo para su posterior consumo
            mlContext.Model.Save(model_km, trainingDataView.Schema, _salida_modelPath);


            //###############################################################
            //VISUALIZACIÓN DEL MODELO
            //###############################################################

            //Definición de las clases de las predicciones: 
            //  -Clase Predicciones: CountryPrediction

            //Inicialización de PlotModel
            var plot = new PlotModel { Title = "Clúster Paises", IsLegendVisible = true };

            //Transformamos el dataset con el Modelo
            var predictionsData = model_km.Transform(trainingDataView);

            //Creamos Array a partir del IDataView y la clase de predicción
            var predictions = mlContext.Data
                .CreateEnumerable<CountryPrediction>(predictionsData, false)
                            .ToArray();

            //Extraemos la lista de los nombres clusteres creados
            var clusters = predictions
                .Select(p => p.PredictedLabel).Distinct().OrderBy(x => x);

            //Construimos el conjunto de puntos para su visualización
            foreach (var cluster in clusters)
            {                 
                var scatter = new ScatterSeries {                    
                    MarkerType = MarkerType.Circle, 
                    MarkerStrokeThickness = 2, 
                    Title = $"Cluster: {cluster}", 
                    RenderInLegend = true };
                //Array ScatterPoint (2 dimensiones) 
                var series = predictions
                    .Where(p => p.PredictedLabel == cluster)
                    //Seleccionamos 2 de las 5 coordenadas de nuestras Features
                    .Select(p => new ScatterPoint(p.Location[2], p.Location[0])).ToArray();
                scatter.Points.AddRange(series);

                plot.Series.Add(scatter);
            }

            //Le damos un color a cada cluster
            plot.DefaultColors = OxyPalettes.HueDistinct(plot.Series.Count).Colors;

            //Guardamos la gráfica en un archivo .svg
            var exporter = new SvgExporter { Width = 1000, Height = 800 };
            using (var fs = new System.IO.FileStream(_salida_plotDataPath, System.IO.FileMode.Create))
            {
                exporter.Export(plot, fs);
            }

            //Guardamos un archivo .csv con el cluster resultante para cada pais
            using (var w = new System.IO.StreamWriter(_salida_ResultDataPath))
            {
                w.WriteLine($"Country;Cluster");
                w.Flush();
                predictions.ToList().ForEach(prediction =>
                {
                    w.WriteLine($"{prediction.country};{prediction.PredictedLabel}");
                    w.Flush();
                });
            }            

        }
    }
}
