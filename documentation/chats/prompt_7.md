# Prompt 7

PII and data privacy:
All PII is contained in our demographics json file.  Therefore, we don't need to search the data.  This can probably be excluded when publishing by excluding all files with "*demographics".  

At the moment, the data is stored in Google Cloud from my Google Workspace organization account.  Only users with specific IAM permissions (i.e., a list of emails with allowed access) can download the data.  For now, only team members can download data.  When we do decide to release the data, we will need a strategy to remove all PII for the downloaded data.  For now, we don't need to check the data itself.  

We need to ensure that no data is uploaded to github.  Rather, all data is acquired from the cloud.  

None of the artifacts will include PII.  Therefore, we can upload data artifacts to github.  However, this will probably clutter the github repo (especially during development).  We probably want to store data artifacts in the cloud.   Team members should be able to download the latest artifacts using our versioning data--which should be included in github.  

Nice-to-haves:
It may be nice to set up GitHub Pages to display reports so team members working on the research report can access it easily.