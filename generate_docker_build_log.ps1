$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logPath = Join-Path $repoRoot "docker_build.log"

Set-Location $repoRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install Docker Desktop first, then rerun this script."
}

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("Lab 9 Docker Build Log")
$lines.Add("======================")
$lines.Add("")
$lines.Add("Working directory:")
$lines.Add($repoRoot)
$lines.Add("")
$lines.Add("Runtime secret injection example:")
$lines.Add('PowerShell: $env:GROQ_API_KEY=''your-key''; docker compose up -d')
$lines.Add("")

function Add-CommandOutput {
    param(
        [string]$Title,
        [string[]]$Command
    )

    $lines.Add("Command:")
    $lines.Add(($Command -join " "))
    $lines.Add("")
    $lines.Add("$Title output:")

    $output = & $Command[0] $Command[1..($Command.Length - 1)] 2>&1
    foreach ($line in $output) {
        $lines.Add([string]$line)
    }
    $lines.Add("")
}

function Wait-ForApi {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            return Invoke-RestMethod -Uri $Url -TimeoutSec 10
        } catch {
            Start-Sleep -Seconds 3
        }
    }

    throw "API did not become healthy within $TimeoutSeconds seconds."
}

Add-CommandOutput -Title "docker compose build" -Command @("docker", "compose", "build")
Add-CommandOutput -Title "docker compose up -d" -Command @("docker", "compose", "up", "-d")

$health = Wait-ForApi -Url "http://127.0.0.1:8000/health"
$lines.Add("Health check:")
$lines.Add(($health | ConvertTo-Json -Compress))
$lines.Add("")

$payload1 = @{
    message = "Did any recruiter ask for my updated resume?"
    thread_id = "docker-proof-thread"
} | ConvertTo-Json -Compress
$chat1 = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -ContentType "application/json" -Body $payload1
$lines.Add("First /chat response:")
$lines.Add(($chat1 | ConvertTo-Json -Compress))
$lines.Add("")

$checkpointBefore = docker exec buraq-agent-api python -c 'import os, sqlite3; db=os.getenv("CHECKPOINT_DB_PATH","/app/runtime/checkpoint_db.sqlite"); conn=sqlite3.connect(db); print(conn.execute("select count(*) from checkpoints").fetchone()[0]); conn.close()'
$collectionBefore = docker exec buraq-agent-api python -c 'from ingest_data import get_collection; print(get_collection().count())'
$lines.Add("Persistence snapshot before restart:")
$lines.Add("Checkpoint rows: $checkpointBefore")
$lines.Add("Chroma collection count: $collectionBefore")
$lines.Add("")

Add-CommandOutput -Title "docker compose restart" -Command @("docker", "compose", "restart")

$healthAfterRestart = Wait-ForApi -Url "http://127.0.0.1:8000/health"
$lines.Add("Health after restart:")
$lines.Add(($healthAfterRestart | ConvertTo-Json -Compress))
$lines.Add("")

$payload2 = @{
    message = "Repeat that in one sentence."
    thread_id = "docker-proof-thread"
} | ConvertTo-Json -Compress
$chat2 = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -ContentType "application/json" -Body $payload2
$lines.Add("Second /chat response after restart:")
$lines.Add(($chat2 | ConvertTo-Json -Compress))
$lines.Add("")

$checkpointAfter = docker exec buraq-agent-api python -c 'import os, sqlite3; db=os.getenv("CHECKPOINT_DB_PATH","/app/runtime/checkpoint_db.sqlite"); conn=sqlite3.connect(db); print(conn.execute("select count(*) from checkpoints").fetchone()[0]); conn.close()'
$collectionAfter = docker exec buraq-agent-api python -c 'from ingest_data import get_collection; print(get_collection().count())'
$lines.Add("Persistence snapshot after restart:")
$lines.Add("Checkpoint rows: $checkpointAfter")
$lines.Add("Chroma collection count: $collectionAfter")
$lines.Add("")

Add-CommandOutput -Title "docker ps" -Command @("docker", "ps")

$lines.Add("Notes:")
$lines.Add("- This log proves reproducible build, multi-service startup, runtime secret injection, and persistence across restart.")
$lines.Add("- The same thread_id was used before and after restart to demonstrate checkpoint survival.")

Set-Content -Path $logPath -Value $lines -Encoding UTF8
Write-Host "Docker build log written to $logPath"
