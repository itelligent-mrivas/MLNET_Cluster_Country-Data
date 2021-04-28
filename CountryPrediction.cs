using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML.Data;


namespace MLNET_Cluster_Country_Data
{
    class CountryPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabel;
        [ColumnName("Score")]
        public float[] Distance;
        [ColumnName("FeaturesNormalized")]
        public float[] Location;
        [ColumnName("country")]
        public string country;
    }
}
