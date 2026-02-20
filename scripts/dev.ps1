param(
  [int]$Port = 8000
)

Write-Host "Starting FastAPI on http://127.0.0.1:$Port"
uvicorn app.main:app --reload --host 127.0.0.1 --port $Port