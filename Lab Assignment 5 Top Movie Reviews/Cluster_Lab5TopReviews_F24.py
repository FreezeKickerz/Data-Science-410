#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 Fall 2024
# # Instructor: Professor John Yen
# # TA: Jin Peng and Al Lawati, Ali Hussain Mohsin
# 
# # Lab 5: Data Frames, SQL Functions, DF-based Join, and Top Movie Reviews 
# 
# # The goals of this lab are for you to be able to
# ## - Use Data Frames in Spark for Processing Structured Data
# ## - Perform Basic DataFrame Transformation: Filtering Rows and Selecting Columns of DataFrame
# ## - Create New Column of DataFrame using `withColumn`
# ## - Use DF SQL Function split to transform a string into an Array
# ## - Filter on a DF Column that is an Array using `array_contains`
# ## - Perform Join on DataFrames 
# ## - Use GroupBy, followed by count and sum DF transformation to calculate the count and the sum of a DF column (e.g., reviews) for each group (e.g., movie).
# ## - Perform sorting on a DataFrame column
# ## - Apply the obove to find Movies in a Genre of your choice that has good reviews with a significant number of ratings (use 10 as the threshold for local mode, 100 as the threshold for cluster mode).
# ## - After completing all exercises in the Notebook, convert the code for processing large reviews dataset and large movies dataset to find movies with top average ranking with at least 100 reviews for a genre of your choice.
# 
# ## Total Number of Exercises: 
# - Exercise 1: 5 points
# - Exercise 2A: 5 points
# - Exercise 2B: 5 points
# - Exercise 3: 5 points
# - Exercise 4: 10 points
# - Exercise 5: 10 points
# - Exercise 6: 5 points
# - Exercise 7: 10 points
# - Exercise 8: 10 points
# - Exercise 9: 10 points
# - Part B (Exercise 10): 25 points (complete spark-submit in the cluster)
# ## Total Points: 100 points
# 
# # Due: midnight, September 30th, 2024
# 

# ## The first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[2]:


import pyspark


# ### Once we import pyspark, we need to import "SparkContext".  Every spark program needs a SparkContext object
# ### In order to use Spark SQL on DataFrames, we also need to import SparkSession from PySpark.SQL

# In[3]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row


# ## We then create a Spark Session variable (rather than Spark Context) in order to use DataFrame. 
# - Note: We temporarily use "local" as the parameter for master in this notebook so that we can test it in ICDS Roar.  However, we need to remove .master("local") before we submit it to Roar to run in cluster mode.

# In[4]:


ss=SparkSession.builder.appName("Lab 5 Top Reviews").getOrCreate()


# In[5]:


ss.sparkContext.setCheckpointDir("~/scratch")


# # Exercise 1 (5 points) 
# - (a) Add your name below AND 
# - (b) replace the path below in both `ss.read.csv` statements with the path of your home directory.
# 
# ## Answer for Exercise 1 (Double click this Markdown cell to fill your name below.)
# - a: Student Name: Tyler Korz

# In[6]:


rating_schema = StructType([ StructField("UserID", IntegerType(), False ),                             StructField("MovieID", IntegerType(), True),                             StructField("Rating", FloatType(), True ),                             StructField("RatingID", IntegerType(), True ),                            ])


# In[7]:


ratings_DF = ss.read.csv("/storage/home/tkk5297/work/Lab5/ratings-large.csv", schema= rating_schema, header=False, inferSchema=False)
# In the cluster mode, we need to change to  `header=False` because it does not have header.


# In[8]:


movie_schema = StructType([ StructField("MovieID", IntegerType(), False),                             StructField("MovieTitle", StringType(), True ),                             StructField("Genres", StringType(), True ),                            ])


# In[9]:


movies_DF = ss.read.csv("/storage/home/tkk5297/work/Lab5/movies-large.csv", schema=movie_schema, header=False, inferSchema=False)
# In the cluster mode, we need to change to `header=False` because it does not have header.


# In[10]:


#movies_DF.printSchema()


# In[11]:


# movies_DF.show(10)


# In[12]:


movies_genres_DF = movies_DF.select("MovieID","Genres")


# In[13]:


movies_genres_rdd = movies_genres_DF.rdd


# In[14]:


#movies_genres_rdd.take(3)


# In[15]:


movies_genres2_rdd = movies_genres_rdd.flatMap(lambda x: x['Genres'].split('|'))


# In[16]:


#movies_genres2_rdd.take(3)


# In[17]:


movies_genres3_rdd = movies_genres2_rdd.map(lambda x: (x, 1))


# In[18]:


movies_genres_count_rdd = movies_genres3_rdd.reduceByKey(lambda x, y: x+y)


# In[19]:


#movies_genres_count_rdd.take(10)


# In[20]:


movies_genres_count_rdd.saveAsTextFile("/storage/home/tkk5297/work/Lab5/MovieGenres_count_cluster.txt")


# In[21]:


#ratings_DF.printSchema()


# In[22]:


#ratings_DF.show(5)


# # 2. DataFrames Transformations
# DataFrame in Spark provides higher-level transformations that are convenient for selecting rows, columns, and for creating new columns.  These transformations are part of Spark SQL.
# 
# ## 2.1 `where` DF Transformation for Filtering/Selecting Rows
# Select rows from a DataFrame (DF) that satisfy a condition.  This is similar to "WHERE" clause in SQL query language.
# - One important difference (compared to SQL) is we need to add `col( ...)` when referring to a column name. 
# - The condition inside `where` transformation can be an equality test, `>` test, or '<' test, as illustrated below.

# # `show` DF action
# The `show` DF action is similar to `take` RDD action. It takes one input parameter, which is the number of elements to be selected from the DF to be displayed.

# # Exercise 2A (5 points) Select a Movie by title
# Complete the following code to select the movie with title "Jurassic Park (1993)".

# In[24]:


#movies_DF.where(col("MovieTitle") == "Jurassic Park (1993)").show()


# In[25]:


#ratings_DF.where(col("Rating") > 3).show(5)


# # `count` DF action
# The `count` action returns the total number of elements in the input DataFrame.

# In[26]:


ratings_DF.filter(4 < col("Rating")).count()


# # Exercise 2B (5 points) Filtering DF Rows
# ### Complete the following statement to (1) select the `ratings_DF` DataFrame for reviews that are exactly 5, and (2) count the total number of such reviews.

# In[27]:


review_5_count = ratings_DF.where(col("Rating") == 5.0).count()
#print(review_5_count)


# ## 2.2 DataFrame Transformation for Selecting Columns
# 
# DataFrame transformation `select` is similar to the projection operation in SQL: it returns a DataFrame that contains all of the columns selected.

# In[28]:


#movies_DF.select("MovieTitle").show(5)


# In[29]:


#movies_DF.select(col("MovieTitle")).show(5)


# # Exercise 3 (5 points) Selecting DF Columns
# ## Complete the following PySpark statement to (1) select only `MovieID` and `Rating` columns, and (2) save it in a DataFrame called `movie_rating_DF`.

# In[30]:


movie_rating_DF = ratings_DF.select(col("MovieID"), col("Rating"))


# In[31]:


#movie_rating_DF.show(5)


# # 2.3 Statistical Summary of Numerical Columns
# DataFrame provides a `describe` method that provides a summary of basic statistical information (e.g., count, mean, standard deviation, min, max) for numerical columns.

# In[32]:


#ratings_DF.describe().show()


# ## RDD has a histogram method to compute the total number of rows in each "bucket".
# The code below selects the Rating column from `ratings_DF`, converts it to an RDD, which maps to extract the rating value for each row, which is used to compute the total number of reviews in 5 buckets.

# In[33]:


ratings_DF.select(col("Rating")).rdd.map(lambda row: row[0]).histogram([0,1,2,3,4,5,6])


# # 3. Transforming the Generes Column into Array of Generes 
# ## We want transform a column Generes, which represent all Generes of a movie using a string that uses "|" to connect the Generes so that we can later filter for movies of a Genere more efficiently.
# ## This transformation can be done using `split` Spark SQL function (which is different from python `split` function)

# In[34]:


Splitted_Generes_DF= movies_DF.select(split(col("Genres"), '\|'))
#Splitted_Generes_DF.show(5)


# ## 3.1 Adding a Column to a DataFrame using withColumn
# 
# # `withColumn` DF Transformation
# 
# We often need to transform content of a column into another column. For example, it is desirable to transform the column Genres in the movies DataFrame into an `Array` of genres that each movie belongs, we can do this using the DataFrame method `withColumn`.

# ### Creates a new column called "Genres_Array", whose values are arrays of genres for each movie, obtained by splitting the column value of "Genres" for each row (movie).

# In[35]:


moviesG2_DF= movies_DF.withColumn("Genres_Array",split("Genres", '\|') )


# In[36]:


#moviesG2_DF.printSchema()


# In[37]:


#moviesG2_DF.show(5)


# # Exercise 4 (10 points)
# Complete the code below to select all movies in a genre of your choice.

# In[40]:


from pyspark.sql.functions import array_contains
movies_your_genre_DF = moviesG2_DF.filter(array_contains("Genres_Array", "Comedy"))


# In[41]:


#movies_your_genre_DF.show(5)


# # An DF-based approach to compute Average Movie Ratings and Total Count of Reviews for each movie.

# # `groupBy` DF transformation
# Takes a column name (string) as the parameter, the transformation groups rows of the DF based on the column.  All rows with the same value for the column is grouped together.  The result of groupBy transformation is often folled by an aggregation across all rows in the same group.  
# 
# # `sum` DF transformation
# Takes a column name (string) as the parameter. This is typically used after `groupBy` DF transformation, `sum` adds the content of the input column of all rows in the same group.
# 
# # `count` DF transformation
# Returns the number of rows in the DataFrame.  When `count` is used after `groupBy`, it returns a DataFrame with a column called "count" that contains the total number of rows for each group generated by the `groupBy`.

# In[42]:


Movie_RatingSum_DF = ratings_DF.groupBy("MovieID").sum("Rating")


# In[43]:


#Movie_RatingSum_DF.show(4)


# # Exercise 5 (5 points)
# Complete the code below to calculate the total number of reviews for each movies.

# In[49]:


Movie_RatingCount_DF = ratings_DF.groupBy("MovieID").count()


# In[50]:


#Movie_RatingCount_DF.show(4)


# # 5. Join Transformation on Two DataFrames

# # Exercise 6 (10 points)
# Complete the code below to (1) perform DF-based inner join on the column MovieID, and (2) calculate the average rating for each movie.

# In[52]:


Movie_Rating_Sum_Count_DF = Movie_RatingSum_DF.join(Movie_RatingCount_DF, "MovieID", 'inner')


# In[53]:


#ovie_Rating_Sum_Count_DF.show(4)


# In[55]:


Movie_Rating_Count_Avg_DF = Movie_Rating_Sum_Count_DF.withColumn("AvgRating", (col("sum(Rating)") / col("count")) )


# In[56]:


#Movie_Rating_Count_Avg_DF.show(4)


# ##  Next, we want to join the avg_rating_total_review_DF with moviesG2_DF

# In[57]:


joined_DF = Movie_Rating_Count_Avg_DF.join(moviesG2_DF,'MovieID', 'inner')


# In[58]:


#moviesG2_DF.printSchema()


# In[59]:


#joined_DF.printSchema()


# In[60]:


#joined_DF.show(4)


# # 6. Filter DataFrame on an Array Column of DataFrame Using `array_contains`
# 
# ## Exercise 7 (10 points)
# Complete the following code to filter for a genre of your choice.

# In[63]:


from pyspark.sql.functions import array_contains
SelectGenreAvgRating_DF = joined_DF.filter(array_contains('Genres_Array',                                                "Comedy")).select("MovieID","AvgRating","count","MovieTitle")


# In[64]:


#SelectGenreAvgRating_DF.show(5)


# In[65]:


SelectGenreAvgRating_DF.count()


# In[66]:


#SelectGenreAvgRating_DF.describe().show()


# In[67]:


SortedSelectGenreAvgRating_DF = SelectGenreAvgRating_DF.orderBy('AvgRating', ascending=False)


# In[68]:


#SortedSelectGenreAvgRating_DF.show(10)


# # Exercise 8 (10 points)
# Use DataFrame method `where` or `filter` to find all movies (in your choice of genre) that have more than 10 reviews (change this to 100 for the cluster mode).

# In[71]:


SortedFilteredSelectGenreAvgRating_DF = SortedSelectGenreAvgRating_DF.where(col("count") > 100)


# In[73]:


#SortedFilteredSelectGenreAvgRating_DF.show(5)


# ## Exercise 9 (10 ponts)
# Complete the code below to save the Movies in your choice of genre, ordered by average rating, that received more than the threshold (10 for local mode, 100 for the cluster mode).  Replace ??? in the file name with the name of your genre.

# In[74]:


output_path = "/storage/home/tkk5297/work/Lab5/Lab5TopReviews/Lab5_SortedFilteredComedyMovieAvgRating_cluster"
SortedFilteredSelectGenreAvgRating_DF.write.csv(output_path)


# In[75]:


ss.stop()


# # Exercise 10 (25 points)
# Enter the following information based on the results of running your code (.py file) on large datasets in the cluster. 
# - Submit your .py file for cluster mode.
# - Submit your log file for a successful execution in the cluster mode.
# - Submit the first file in your output directory (generated by Exercise 9 in the cluster mode).

# - What is your choice of the genre for your analysis? 
# - What are the top five movies in the genre?
# - What is the computation time your job took? 
