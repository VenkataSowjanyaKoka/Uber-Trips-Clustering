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
