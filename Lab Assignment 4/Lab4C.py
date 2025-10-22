#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 Fall 2024
# ## Instructor: Professor John Yen
# ## TAs:Jin Peng and Al Lawati, Ali Hussain Mohsin
# 
# # Lab 3: Hashtag Counting and Spark-submit in Cluster Mode
# ## The goals of this lab are for you to be able to
# ## - Use the RDD transformations ``filter`` and ``sortBy``.
# ## - Compute hashtag counts for an input data file containing tweets.
# ## - Apply the obove to compute hashtag counts for tweets related to Boston Marathon Bombing (gathered on April 17, 2013, two days after the domestic terrorist attack).
# 
# ## Total Number of Exercises: 7
# - Exercise 1: 5 points
# - Exercise 2: 10 points
# - Exercise 3: 10 points
# - Exercise 4: 5 points
# - Exercise 5: 10 points
# - Exercise 6: 5 points
# - Exercise 7: 10 points6
# 
# ## Total Points: 55 points
# 
# ## Data for Lab 3
# - sampled_BMB_4_17_tweets.csv : A random sampled of a small set of tweets regarding Boston Marathon Bombing on April 17, 2013. This data is used in the local mode.
# - Like Lab2, download the data from Canvas into a directory for the lab (e.g., Lab3) under your home directory.
# 
# ## Items to submit for Lab 3
# - Completed Jupyter Notebook (HTML format)
# - The first and second output file in your output directory
# - a screen shot of the ``ls -al`` command in the output directory for a successful run in the cluster mode.
# 
# # Due: midnight, midnight Sep 17, 2024

# ## Like Lab 2, the first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[1]:


import pyspark


# In[2]:


from pyspark import SparkContext


# ## Like Lab 2, create a Spark Context object.  
# 
# - Note: We use "local" as the master parameter for ``SparkContext`` in this notebook so that we can run and debug it in ICDS Jupyter Server.  However, we need to remove ``"master="local",``later when you convert this notebook into a .py file for running it in the cluster mode.

# In[3]:


sc=SparkContext(appName="Lab3")
# sc


# In[4]:


sc.setLogLevel("WARN")


# # Exercise 1 (5 points)  Add your name below 
# ## Answer for Exercise 1
# - Your Name: Tyler Korz

# # Exercise 2 (10 points) 
# ## Complete the path and run the code below to read the file "sampled_BMB_4_17_tweets.csv" from your Lab3 directory.

# In[5]:


tweets_RDD = sc.textFile("/storage/home/tkk5297/work/Lab4/BMB_4_17_tweets.csv")
# tweets_RDD


# # Exercise 3 (10 points) 
# ## Complete and execute the code below, which computes the total count of hashtags in the input tweets.
# - (a) Uses flatMap to "flatten" the list of tokens from each tweet (using split function) into a very large list of tokens.
# - (b) Filter the token for hashtags.
# - (c) Count the total number of hashtags in a way similar to Lab 2.

# ## Code for Exercise 3 is shown in the Code Cells below.

# In[6]:


tokens_RDD = tweets_RDD.flatMap(lambda line: line.strip().split(" "))
tokens_RDD.take(3)


# # take (action for RDD)
# - ``take`` is an action for RDD.  
# - The parameter is the number of elements from the input RDD you want to show.
# - `take` is often used for debugging/learning purpose in the local mode so that the contents of a few samples of an RDD can be revealed.  This way, if the content and/or the format of the RDD differs from what you expected, you can investigate it and, if needed, fix it before proceeding further.

# # filter (transformation for RDD)
# 
# - The syntax for filtering (one type of data trasnformation in Spark) an input RDD is
# ``<input RDD>.filter(lambda <parameter> : <the body of a Boolean function> )``
# - Notice the syntax is not what is described in p. 38 of the textbook.
# - The result of filtering the input RDD is the collection of all elements in the input RDD that pass the filter condition (i.e., returns True when the filtering Boolean function is applied to each element of the input RDD). 
# - For example, the filtering condition in the pyspark conde below checks whether each element of the input RDD (i.e., `tokens_RDD`) starts with the character "#", using Python `startswith()` method for string.

# In[7]:


hashtag_RDD = tokens_RDD.filter(lambda x: x.startswith("#"))


# In[8]:


hashtag_1_RDD = hashtag_RDD.map(lambda x: (x, 1))


# In[9]:


hashtag_count_RDD = hashtag_1_RDD.reduceByKey(lambda x, y: x+y, 5)


# # Exercise 4 (5 points)
# Use take(n) to show the first 5 key-value pairs (hashtag, count) in hashtag_count_RDD.

# In[10]:


hashtag_count_RDD.take(5)


# # sortBy (transformation for RDD)
# - To sort hashtag count so that those that occur more frequent appear first, we use ``sortBy(lambda pair: pair[1], ascending=False)``.
# - `sortBy` sort the input RDD based on the value of the lambda function, which returns the value of the input key-value pair.  
# - Note: The index of a list/tuple in Python starts with 0. Therefore `pair[0]` accesses the key of each key-value pair (in the input RDD), whereas `pair[1]` accesses the value of the key-value pair in the input RDD.
# - The default sorting order is ascending. To sort in descending order, we need to set the parameter `ascending` to `False`, which means frequent/top hashtags occured first in the output.

# # Exercise 5 (10 points) 
# ## Complete and execute the code below, which sort hashtag count by count (in descending order)
# - Note: Sort the hashtag count, which occurs in the value position, in descending order.

# In[11]:


sorted_hashtag_count_RDD = hashtag_count_RDD.sortBy(lambda pair: pair[1], ascending=False)


# # Exercise 6 (5 points)
# Use take on sorted_hashtag_count_RDD to show the top 10 hashtags, based on their counts.

# In[12]:


sorted_hashtag_count_RDD.take(10)


# # Exercise 7 (10 points)
# Use saveAsTextFile to save your count of hashtag counts.

# ### Note: You need to complete the path with your output directory (e.g., Lab3 under your work directory). 

# In[13]:


output_path = "/storage/home/tkk5297/work/Lab4/sorted_hashtag_count_cluster.txt" 
sorted_hashtag_count_RDD.saveAsTextFile(output_path)


# In[14]:


sc.stop()


# In[ ]:





# In[ ]:




