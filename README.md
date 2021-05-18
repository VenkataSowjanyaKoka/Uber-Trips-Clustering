# Uber-Trips-Clustering
Clustering in PySpark

# Introduction:
The New York City Taxi and Limousine Commission (NYC TLC) maintains trip data on taxi service and For-Hire-Vehicles (FHV, including Uber and Lyft) to improve safety, accountability, and guide policymaking. This data typically includes timestamp (date and time) of pickup, pickup and drop-off location, and company that provided the vehicle for passenger pickup.  For Uber trips, NYC TLC has made data on over 4.5 million Uber pickups in the New York City from April through September 2014 available for pattern learning purposes. The Uber pickup data contains data on the following fields:
1.	Timestamp of pickup
2.	Latitude of pickup location
3.	Longitude of pickup location
4.	TLC base company code affiliated with Uber pickup. This the TLC code for the company that provided the vehicle for pickup. The base codes are for the following Uber bases:
a.	B02512	- Unter LLC
b.	B02598	- Hinter LLC
c.	B02617	- Weiter
d.	B02682	- Schmecken
e.	B02764	- Danach-NY
f.	B02765	- Grun
g.	B02835	- Dreist
h.	B02836	- Drinnen

In this project we will work with the Uber trip data from April 2014. There were approximately 565,000 recorded Uber trips between 31-March-2014 and 30-April-2014, with 5 base companies providing the vehicles. The TLC codes for these five companies are B02512, B02598, B02617, B02682, and B02764. 

The objective is to determine the best K-Means Clustering model to see if trips based on longitude and latitude of pickup location are assigned to cluster corresponding to the company that provided the vehicle.

# Steps followed and questions answered,

Questions to be Answered (25 points)
1.	What is the shape of the data contained in the UberApril14.CSV?
2.	How many Uber trips were recorded for each company (by base code). What can you say about the distribution trips among companies? Are there companies that dominate in terms of the percentage share of the trips? 
3.	What features (or attributes) are recorded for each Uber trip? Does any attribute require transformation because of data type requirements in Clustering? If so, identify the feature and comment on the type of transformation required. Include these comments in your notebook.
4.	Perform the transformations, if any, identified in step # 3. Perform feature engineering if and where needed, including vectorization of relevant variables. Provide a printout of the schema of your feature-engineered data.
5.	To train and then test your model, split the data from UberApril14 into training and test datasets using a 75/25 split. Like you did in step 2 above, comment on the percentage distribution of trips among companies in both the training and test datasets. Are they representative of the overall data? Include your answer as comments in the notebook.
6.	Build and train KMeans Clustering model. For this you, will use the Elbow method to identify the number of clusters to start the algorithm. Use a seed value to ensure each iteration starts with the same initial set of conditions. Experiment with (n-1) and (n+1) number of clusters, where n is the optimal number found by the Elbow method. For each run, generate the SSE and Silhouette Coefficient. Select the best model on the basis of SSE and Silhouette Coefficient. 
7.	Using the best trained model from step 6, test the performance of the model against the test dataset. Again, measure the performance by computing the SSE and Silhouette Coefficient. Comment of the accuracy of the clustering model on the basis these metrics as well as by comparing the distribution of trips among companies in the original dataset. 
8.	Do your own research on evaluation metrics, other than the Silhouette Coefficient, that may be used to measure the performance of the KMeans Clustering algorithm as implemented by pyspark’s MLlib in python. You may want to see how your final model performs on these other metrics. 

