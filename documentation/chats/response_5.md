# Claude Response 5

Excellent point! Let's design a comprehensive artifact management system that cleanly separates ETL metadata from EDA outputs while enabling unified reporting. This is crucial for both debugging and understanding the data pipeline.
Artifact Management Architecture
1. Artifact Types and Structure
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
2. Artifact Manager
python# utils/artifact_manager.py
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

class ArtifactType(Enum):
    ETL_METADATA = "etl_metadata"
    EDA_REPORT = "eda_report"
    PLOT = "plot"
    TABLE = "table"
    LOG = "log"
    SUMMARY = "summary"

@dataclass
class Artifact:
    """Single artifact metadata"""
    artifact_id: str
    artifact_type: ArtifactType
    stage: str
    name: str
    description: str
    file_path: Path
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['artifact_type'] = self.artifact_type.value
        d['file_path'] = str(self.file_path)
        d['created_at'] = self.created_at.isoformat()
        return d

class ArtifactManager:
    """Centralized artifact management"""
    
    def __init__(self, version_id: str, base_dir: Path = Path("artifacts")):
        self.version_id = version_id
        self.base_dir = base_dir / version_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_file = self.base_dir / "artifact_manifest.json"
        self.artifacts: Dict[str, Artifact] = {}
        self._load_manifest()
    
    def register_artifact(self, 
                         artifact_type: ArtifactType,
                         stage: str,
                         name: str,
                         file_path: Path,
                         description: str = "",
                         metadata: Optional[Dict] = None) -> str:
        """Register a new artifact"""
        artifact_id = f"{stage}_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        artifact = Artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            stage=stage,
            name=name,
            description=description,
            file_path=file_path,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.artifacts[artifact_id] = artifact
        self._save_manifest()
        
        return artifact_id
    
    def get_artifacts_by_stage(self, stage: str) -> List[Artifact]:
        """Get all artifacts for a specific stage"""
        return [a for a in self.artifacts.values() if a.stage == stage]
    
    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """Get all artifacts of a specific type"""
        return [a for a in self.artifacts.values() if a.artifact_type == artifact_type]
    
    def save_dataframe(self, df: pd.DataFrame, stage: str, name: str, 
                      description: str = "") -> str:
        """Save a DataFrame as an artifact"""
        file_path = self.base_dir / "tables" / f"{stage}_{name}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(file_path)
        
        return self.register_artifact(
            ArtifactType.TABLE,
            stage,
            name,
            file_path,
            description,
            metadata={
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
        )
    
    def save_plot(self, fig, stage: str, name: str, 
                  description: str = "", dpi: int = 300) -> str:
        """Save a matplotlib figure as an artifact"""
        file_path = self.base_dir / "plots" / f"{stage}_{name}.png"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        
        return self.register_artifact(
            ArtifactType.PLOT,
            stage,
            name,
            file_path,
            description
        )
3. ETL Metadata Collector
python# utils/etl_metadata.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

@dataclass
class CleaningMetadata:
    """Metadata from cleaning stage"""
    total_records: int
    valid_records: int
    invalid_records: int
    outliers_removed: int
    duplicates_removed: int
    missing_data_handled: Dict[str, int]
    validation_errors: List[Dict[str, Any]]
    processing_time: float
    
    def to_summary(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "data_quality_score": self.valid_records / self.total_records * 100,
            "issues_found": len(self.validation_errors),
            "outliers_removed": self.outliers_removed
        }

@dataclass
class ExtractionMetadata:
    """Metadata from feature extraction"""
    input_records: int
    output_records: int
    features_extracted: List[str]
    extraction_errors: List[Dict[str, Any]]
    computation_time: float
    memory_usage_mb: float
    
class ETLMetadataCollector:
    """Collect and manage ETL metadata"""
    
    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager
        self.metadata: Dict[str, Any] = {}
    
    def collect_cleaning_metadata(self, stage_name: str, 
                                 metadata: CleaningMetadata,
                                 outliers_df: Optional[pd.DataFrame] = None):
        """Collect metadata from cleaning stage"""
        # Save summary
        summary_path = self.artifact_manager.base_dir / "etl_metadata" / stage_name / "cleaning_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(metadata.to_summary(), f, indent=2)
        
        self.artifact_manager.register_artifact(
            ArtifactType.ETL_METADATA,
            stage_name,
            "cleaning_summary",
            summary_path,
            "Summary of data cleaning process"
        )
        
        # Save detailed outliers if provided
        if outliers_df is not None and not outliers_df.empty:
            self.artifact_manager.save_dataframe(
                outliers_df,
                stage_name,
                "outliers_removed",
                "Details of outliers removed during cleaning"
            )
        
        # Save validation errors
        if metadata.validation_errors:
            errors_path = summary_path.parent / "validation_errors.json"
            with open(errors_path, 'w') as f:
                json.dump(metadata.validation_errors, f, indent=2)
            
            self.artifact_manager.register_artifact(
                ArtifactType.ETL_METADATA,
                stage_name,
                "validation_errors",
                errors_path,
                "Validation errors encountered"
            )
4. Updated Pipeline Stages with Metadata Collection
python# scripts/pipeline/02_clean_data.py (updated)
from utils.artifact_manager import ArtifactManager
from utils.etl_metadata import ETLMetadataCollector, CleaningMetadata

def clean_data(version_id: str, config: Dict[str, Any], 
               artifact_manager: ArtifactManager) -> Path:
    """Clean data with metadata collection"""
    
    start_time = time.time()
    metadata_collector = ETLMetadataCollector(artifact_manager)
    
    # Load raw data
    raw_data = pd.read_parquet(input_path)
    initial_count = len(raw_data)
    
    # Track what we remove
    outliers_removed = pd.DataFrame()
    validation_errors = []
    
    # Cleaning operations with tracking
    # 1. Remove outliers
    outlier_mask = detect_outliers(raw_data)
    outliers_removed = raw_data[outlier_mask]
    cleaned_data = raw_data[~outlier_mask]
    
    # 2. Handle missing data
    missing_before = cleaned_data.isnull().sum().to_dict()
    cleaned_data = handle_missing_data(cleaned_data)
    missing_after = cleaned_data.isnull().sum().to_dict()
    
    # 3. Validate data
    validation_results = validate_data(cleaned_data)
    validation_errors = validation_results['errors']
    
    # Collect metadata
    cleaning_metadata = CleaningMetadata(
        total_records=initial_count,
        valid_records=len(cleaned_data),
        invalid_records=initial_count - len(cleaned_data),
        outliers_removed=len(outliers_removed),
        duplicates_removed=0,  # Add your logic
        missing_data_handled={
            col: missing_before[col] - missing_after[col] 
            for col in missing_before
        },
        validation_errors=validation_errors,
        processing_time=time.time() - start_time
    )
    
    # Save metadata
    metadata_collector.collect_cleaning_metadata(
        "cleaned_data",
        cleaning_metadata,
        outliers_removed
    )
    
    # Save cleaned data
    output_path = save_cleaned_data(cleaned_data)
    
    return output_path
5. Unified Report Generator
python# utils/report_generator.py
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UnifiedReportGenerator:
    """Generate unified reports from all artifacts"""
    
    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager
        self.template_env = Environment(
            loader=FileSystemLoader('templates/reports')
        )
    
    def generate_pipeline_summary(self) -> Path:
        """Generate comprehensive pipeline summary"""
        # Collect all ETL metadata
        etl_artifacts = self.artifact_manager.get_artifacts_by_type(
            ArtifactType.ETL_METADATA
        )
        
        # Parse metadata
        pipeline_stats = self._aggregate_pipeline_stats(etl_artifacts)
        
        # Create visualizations
        figures = self._create_pipeline_visualizations(pipeline_stats)
        
        # Generate HTML report
        template = self.template_env.get_template('pipeline_summary.html')
        html_content = template.render(
            version_id=self.artifact_manager.version_id,
            stats=pipeline_stats,
            figures=figures,
            artifacts=etl_artifacts
        )
        
        output_path = self.artifact_manager.base_dir / "unified_reports" / "pipeline_summary.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)
        
        return output_path
    
    def generate_data_quality_dashboard(self) -> Path:
        """Interactive data quality dashboard"""
        # Collect all relevant artifacts
        cleaning_artifacts = [
            a for a in self.artifact_manager.artifacts.values()
            if 'cleaning' in a.stage or 'quality' in a.name
        ]
        
        # Create interactive dashboard with Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Records Processed', 'Data Quality Score', 
                          'Issues by Stage', 'Processing Time'),
            specs=[[{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Add visualizations
        # ... (dashboard creation logic)
        
        # Save as HTML with Plotly
        output_path = self.artifact_manager.base_dir / "unified_reports" / "data_quality_dashboard.html"
        fig.write_html(output_path, include_plotlyjs='cdn')
        
        return output_path
    
    def generate_full_report(self, include_eda: bool = True) -> Path:
        """Generate comprehensive PDF report"""
        # This would use a library like ReportLab or WeasyPrint
        # to create a professional PDF report
        pass
6. Report Templates
html<!-- templates/reports/pipeline_summary.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Summary - {{ version_id }}</title>
    <style>
        /* Modern, professional styling */
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .metric-card { 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 10px;
        }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .stage-section { margin: 30px 0; }
        .artifact-link { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipeline Execution Summary</h1>
        <p>Version: {{ version_id }}</p>
        <p>Generated: {{ now() }}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Records Processed</h3>
                <div class="metric-value">{{ stats.total_records }}</div>
            </div>
            <div class="metric-card">
                <h3>Data Quality Score</h3>
                <div class="metric-value">{{ stats.quality_score }}%</div>
            </div>
            <div class="metric-card">
                <h3>Processing Time</h3>
                <div class="metric-value">{{ stats.total_time }}min</div>
            </div>
        </div>
        
        {% for stage in stats.stages %}
        <div class="stage-section">
            <h2>{{ stage.name }}</h2>
            <table>
                <tr><td>Records In:</td><td>{{ stage.records_in }}</td></tr>
                <tr><td>Records Out:</td><td>{{ stage.records_out }}</td></tr>
                <tr><td>Issues:</td><td>{{ stage.issues }}</td></tr>
                <tr><td>Time:</td><td>{{ stage.time }}s</td></tr>
            </table>
            
            <h4>Artifacts</h4>
            <ul>
            {% for artifact in stage.artifacts %}
                <li>
                    <a href="{{ artifact.file_path }}" class="artifact-link">
                        {{ artifact.name }}
                    </a> - {{ artifact.description }}
                </li>
            {% endfor %}
            </ul>
        </div>
        {% endfor %}
    </div>
</body>
</html>
7. Integration with Pipeline
python# scripts/pipeline/run_pipeline.py (updated)
@click.option('--generate-reports', is_flag=True, 
              help='Generate unified reports after pipeline')
def run_pipeline(..., generate_reports: bool):
    """Pipeline with integrated reporting"""
    
    # Initialize artifact manager
    artifact_manager = ArtifactManager(version_id)
    
    # Run pipeline stages with artifact collection
    stages_config = {
        'download': {'func': download_data, 'args': [artifact_manager]},
        'clean': {'func': clean_data, 'args': [artifact_manager]},
        'keypairs': {'func': extract_keypairs, 'args': [artifact_manager]},
        'features': {'func': extract_features, 'args': [artifact_manager]}
    }
    
    for stage_name, stage_config in stages_config.items():
        if stage_name in stages_to_run:
            stage_config['func'](version_id, config, *stage_config['args'])
    
    # Generate reports if requested
    if generate_reports:
        report_gen = UnifiedReportGenerator(artifact_manager)
        
        # Generate all report types
        reports = {
            'pipeline_summary': report_gen.generate_pipeline_summary(),
            'data_quality': report_gen.generate_data_quality_dashboard()
        }
        
        if not skip_eda:
            reports['full_report'] = report_gen.generate_full_report(include_eda=True)
        
        # Display report locations
        click.echo("\nGenerated Reports:")
        for name, path in reports.items():
            click.echo(f"  {name}: {path}")
8. Usage Examples
bash# Run pipeline with full artifact collection and reporting
python scripts/pipeline/run_pipeline.py \
    --mode full \
    --generate-reports

# Generate reports for existing version
python scripts/standalone/generate_reports.py \
    --version-id 2024-01-15_14-30-00_macbook-alice \
    --report-types pipeline_summary data_quality

# Access specific artifacts programmatically
python -c "
from utils.artifact_manager import ArtifactManager
am = ArtifactManager('current')
cleaning_artifacts = am.get_artifacts_by_stage('cleaned_data')
for a in cleaning_artifacts:
    print(f'{a.name}: {a.file_path}')
"
Key Benefits

Complete Separation: ETL metadata never mixed with EDA outputs
Unified Access: Single interface to all artifacts
Automatic Reports: Generate comprehensive reports with one command
Traceability: Every number in reports traceable to source artifact
Flexible Output: HTML dashboards, PDFs, interactive notebooks
Debugging Support: Easy to investigate issues with detailed metadata
Extensible: Easy to add new artifact types or report formats

Would you like me to elaborate on any specific component or show more implementation details?