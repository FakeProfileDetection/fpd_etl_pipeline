artifacts/
├── {version_id}/
│   ├── etl_metadata/              # ETL process artifacts
│   │   ├── download/
│   │   │   ├── manifest.json      # What was downloaded
│   │   │   └── download_log.json  # Download details
│   │   ├── cleaning/
│   │   │   ├── cleaning_report.json
│   │   │   ├── outliers_removed.csv
│   │   │   ├── validation_errors.json
│   │   │   └── cleaning_stats.json
│   │   ├── keypairs/
│   │   │   ├── extraction_stats.json
│   │   │   ├── invalid_sequences.csv
│   │   │   └── processing_log.json
│   │   └── features/
│   │       ├── {feature_type}/
│   │       │   ├── extraction_stats.json
│   │       │   ├── feature_summary.json
│   │       │   └── validation_report.json
│   │       └── feature_registry.json
│   ├── eda_reports/               # EDA outputs
│   │   ├── data_quality/
│   │   │   ├── report.html
│   │   │   ├── figures/
│   │   │   └── tables/
│   │   └── {report_name}/
│   │       └── ...
│   ├── unified_reports/           # Combined reports
│   │   ├── pipeline_summary.html
│   │   ├── data_quality_dashboard.html
│   │   └── full_report.pdf
│   └── artifact_manifest.json     # Master catalog

