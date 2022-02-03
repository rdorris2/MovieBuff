#!/bin/bash

mkdir datasets
cd datasets
wget https://github.com/rdorris2/MovieBuff/tree/master/data/the-movies-dataset.zip
unzip the-movies-dataset.zip
rm the-movies-dataset.zip

------------------------------

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType,IntegerType
spark = SparkSession.builder.appName('Names').getOrCreate()
sc = spark.sparkContext

movies = spark.read.csv("data/the-movies-dataset/movies_metadata.csv",header=True)
credits = spark.read.csv("datasets/the-movies-dataset/credits.csv",header=True)

movies_credits = movies.joing(credits, ["movieId"], "left")

names = movies_credits.rdd.map(lambda x: \
    (x.cast,x.cast["gender"],x.cast["name"])) \
    .toDF(["cast","gender","name"])


-----------------------------------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Recommendations').getOrCreate()

credits = spark.read.csv("data/the-movies-dataset/credits.csv",header=True)

from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext
import sys

sc = SparkContext()
sc.setLogLevel("ERROR")

def predict(movieID):

    completeMovies = sc.textFile('data/the-movies-dataset/movies_metadata.csv')
    header2 = completeMovies.first()
    completeMovies = completeMovies.filter(lambda line : line != header2)\
        .map(lambda line : line.split(","))
    

    model = MatrixFactorizationModel.load(sc, "target/model")

    completeRDD = sc.textFile('datasets/the-movies-dataset/keywords.csv')
    header = completeRDD.first()

    completeRDD = completeRDD.filter(lambda line : line != header)\
    .map(lambda line : line.split(","))\
    .map(lambda line : (line[0],line[1],line[2]))

    keywords = completeRDD.filter(lambda line : line[0] == ["friendship"]).map(lambda line : line[1]).collect()

    nonKeywords = completeMovies.filter(lambda line : line[0] not in keywords).map(lambda line : (movieId,line[0]))

    predict = model.predictAll(nonKeywords).map(lambda line : [str(line[1]), line[2]]).sortBy(lambda line : line[1], ascending=False)

    movies = predict.join(completeMovies)

    output = movies.map(lambda line : line[1][1]).take(15)

    for name in output:
        print(name)


if __name__ == '__main__':
    userID = sys.argv[1]
    predict(movieId)

