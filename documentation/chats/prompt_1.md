# Prompt 1

You are an expert at ETL and EDA pipelines and data analysis. Your task is to help me clean up my scripts, modifying as need to work with the new data collection and ETL pipeline.

Context:
Our team is switching to a new dataset which is collected via our web app. Our current eda repository  contains scripts used on the old data, some of which are still valid and re-usable, others need to be cleaned.  We also want an end-to-end pipeline that runs with a single script.  But we also want individual scripts so we can work on various sub-stages of the pipeline.

Use cases:
* A team members wants to download the most recent web app data and re-run the pipeline.  They also want to make the artifacts available to other team members.  We affix the same timestamp-hostname to every product from a particular download and processing of web app data so team members can identify the most recent dataset and who processed it.
* A team members want to download another team member’s web-app data or processed data from any stage of the pipeline.  I’m considering a .env.current environment file that both tracks all versions and adds the most current version to the end of the file—ensuring that is sourced last.  There could be a better way.  What are best options.  (Team members should be able to run a single script after their environment is set up.)
* A team member wants to work on the pipeline itself, not saving any artifacts to the cloud.  This is for development. 

I envision multiple scripts so the pipeline development team can work on the scripts at various stages.  Also, I envision a single script pipeline that can run end-to-end on a new set of data and a script that can download all artifact for a current dataset version (i.e., no processing, just get all the data).

Timeline:
* I’ve got about 5 hours to build a prototype to test on our current data.  (This needs to verify our web app data is correct before we begin a full data collection campaign.)

My team:
* Data science professor at a major university.
* PhD candidate in data science
* Software engineer learning machine learning and data science
* ME: I am a machine learning engineer with extensive Python experience. Most of my projects are data science and applied machine learning research.

I have access to Claude and other Chat bots to speed up coding.  I also have CoPilot in VS Code—still learning everything I can do with it.

Hardware:
* Most team members are working either on a Mac or Linux machine.
* ME: I’m working on both a Mac Book (m1) and a remote Linux machine.

Current thoughts (is this the best option?)
Before we code, let’s think about design.