using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML.Data;

namespace MLNET_Cluster_Country_Data
{
    public class CountryObservation
    {
        //#####################################################################################################################################
        //WEB DATOS: https://www.kaggle.com/rohan0301/unsupervised-learning-on-country-data
        //#####################################################################################################################################

        //Definición del nombre y del dominio de los atributos en relación con las columnas de la tabla de obs
        //No es necesario cargar todos los atributos de la tabla, solo los que se vayan a utilizar

        [LoadColumn(0)]
        public string country;

        [LoadColumn(1)]
        public float child_mort;

        [LoadColumn(2)]
        public float exports;

        [LoadColumn(3)]
        public float health;

        [LoadColumn(4)]
        public float imports;

        [LoadColumn(5)]
        public float income;

        [LoadColumn(6)]
        public float inflation;

        [LoadColumn(7)]
        public float life_expec;

        [LoadColumn(8)]
        public float total_fer;

        [LoadColumn(9)]
        public float gdpp;        
    }
    
}
