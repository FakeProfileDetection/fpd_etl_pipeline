# Prompt 3

Assume:
* web app data never changes.  We only add or delete data from the Google Cloud bucket.  This data comes from our web app.  
* We always run clean_data on a newly downloaded dataset.  This prepares our data for all downstream tasks.  If we change how the data is cleaned, we may or may not want to download new web app data.  We will typically only download new web app data if we anticipate new data has been collected by the web app.  
* We almost always start all feature extraction from either the clean_data or from the extracted keypairs. 
* During development, we may want to force re-extraction or re-clean the data on the same downloaed dataset.  Also, we may want to re-extract some of the features.  

How does your system handle this?