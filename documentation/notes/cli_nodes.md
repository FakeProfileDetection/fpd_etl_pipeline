 The pipeline now shows:
  ☁️  Cloud mode: Will download latest data from cloud storage
  📤 Upload enabled: Results will be uploaded to cloud after processing

  Usage Examples

  Standard run (downloads from cloud, processes, saves locally):
  python scripts/pipeline/run_pipeline.py --mode full

  Development mode (uses local data):
  python scripts/pipeline/run_pipeline.py --mode full --local-only

  Process and upload results:
  python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts

  Upload existing results later:
  python scripts/standalone/upload_artifacts.py --version-id 2025-06-29_21-23-04_loris-mbpcablercncom

  The new design follows the principle of least surprise - default behavior is what non-technical users expect (download latest data),
  while developers have clear flags for their specific needs.

