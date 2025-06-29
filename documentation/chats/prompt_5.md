# Prompt 5

Let's discuss one more thing before we document and start implementing.

My current scripts perform some eda as well as extract detailed information about cleaning and extraction (for example, number of outliers, number of valid data, any issues during cleaning or extraction).  We'll need to separate these so we keep the etl cleaning and extraction artifacts separate from the eda artifacts.  Nevertheless, we'll ultimately want to store all artifacts as well as generate a comprehensive report on some or parts of the artifacts.  How will your pipeline manage the separate artifacts in a way that reports (i.e., interactive html, downloadable plots, summary pds, etc) can be generated automatically?