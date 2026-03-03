# ============================================================
# Configuration
# ============================================================
$verbose = $false       # Set to $true for detailed logging
$minutesBack = 0       # Only include files modified within this many minutes (0 = all files)

# Folders to exclude entirely
$excludeFolders = @(
    'models', 
    'venv',
    'env', 
    'output', 
    'temp', 
    '__pycache__', 
    '.git', 
    'saved_configs',
    'insightface',
    'codeformer',
    'gfpgan',
    'checkpoints',
    'weights'
)

# File extensions to exclude
$excludeExtensions = @(
    '.onnx', '.pth', '.pt', '.bin', '.pkl',
    '.mp4', '.avi', '.mkv', '.webm', '.gif',
    '.jpg', '.jpeg', '.png', '.webp', '.bmp',
    '.zip', '.tar', '.gz', '.7z',
    '.npy', '.npz',
    '.safetensors', '.ckpt',
    '.db', '.sqlite',
    '.exe', '.dll', '.so', '.pyd'
)

# ============================================================
# Setup
# ============================================================
$source = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stagingDir = "$PSScriptRoot\..\roop-claude-upload_$timestamp"
$logFile = "$PSScriptRoot\..\roop-claude-upload_$timestamp.log"
$cutoffTime = (Get-Date).AddMinutes(-$minutesBack)

function Write-Log {
    param([string]$message, [string]$level = "INFO")
    $entry = "[$(Get-Date -Format 'HH:mm:ss')] [$level] $message"
    if ($verbose) {
        switch ($level) {
            "INFO"    { Write-Host $entry -ForegroundColor Cyan }
            "INCLUDE" { Write-Host $entry -ForegroundColor Green }
            "EXCLUDE" { Write-Host $entry -ForegroundColor Yellow }
            "WARN"    { Write-Host $entry -ForegroundColor Magenta }
            "ERROR"   { Write-Host $entry -ForegroundColor Red }
            default   { Write-Host $entry }
        }
    } else {
        if ($level -eq "INFO" -or $level -eq "ERROR" -or $level -eq "WARN") {
            Write-Host $entry
        }
    }
    Add-Content -Path $logFile -Value $entry
}

# ============================================================
# Main
# ============================================================
Write-Log "Export started"
Write-Log "Source      : $source"
Write-Log "Staging dir : $stagingDir"
if ($minutesBack -gt 0) {
    Write-Log "Changed since: $($cutoffTime.ToString('HH:mm:ss')) (last $minutesBack minutes)"
} else {
    Write-Log "Changed since: ALL FILES (minutesBack = 0)"
}

New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null

$includedFiles = @()
$excludedCount = 0
$allFiles = Get-ChildItem -Path $source -Recurse -File

foreach ($file in $allFiles) {
    $reason = $null
    $relativePath = $file.FullName.Substring($source.Length + 1).Replace('\', '/')

    # Check excluded folders
    foreach ($folder in $excludeFolders) {
        if ($file.FullName -like "*\$folder\*" -or $file.FullName -like "*\$folder") {
            $reason = "excluded folder '$folder'"
            break
        }
    }

    # Check excluded extensions
    if (-not $reason -and ($excludeExtensions -contains $file.Extension.ToLower())) {
        $reason = "excluded extension '$($file.Extension)'"
    }

    # Check file size
    if (-not $reason -and $file.Length -gt 50MB) {
        $reason = "file too large ($([math]::Round($file.Length / 1MB, 1)) MB)"
    }

    # Check modified time (skip check if minutesBack = 0, meaning include all)
    if (-not $reason -and $minutesBack -gt 0 -and $file.LastWriteTime -lt $cutoffTime) {
        $reason = "not recently modified (last modified $($file.LastWriteTime.ToString('HH:mm:ss')))"
    }

    if ($reason) {
        Write-Log "SKIP : $relativePath -- $reason" "EXCLUDE"
        $excludedCount++
    } else {
        Write-Log "INCLUDE : $relativePath" "INCLUDE"
        $includedFiles += $file
    }
}

Write-Log "Scan complete -- $($includedFiles.Count) included, $excludedCount excluded" "INFO"

if ($includedFiles.Count -eq 0) {
    Write-Log "No files matched -- try increasing minutesBack or set it to 0 for all files." "WARN"
    Start-Sleep -Seconds 5
    exit
}

# ============================================================
# Copy files into flat staging folder, prefixing path into filename
# e.g. app/roop/core.py becomes app__roop__core.py
# This lets Claude see the full path context from the filename alone
# ============================================================
foreach ($file in $includedFiles) {
    $relativePath = $file.FullName.Substring($source.Length + 1).Replace('\', '/')
    $flatName = $relativePath.Replace('/', '__')
    $destPath = Join-Path $stagingDir $flatName
    Copy-Item -Path $file.FullName -Destination $destPath
    Write-Log "Copied: $relativePath -> $flatName" "INCLUDE"
}

Write-Log "" "INFO"
Write-Log "Done! $($includedFiles.Count) file(s) exported to:" "INFO"
Write-Log "$stagingDir" "INFO"
Write-Log "" "INFO"
Write-Log "Next step: select all files in that folder and drag them into Claude." "INFO"
Write-Log "Log saved to: $logFile" "INFO"

# Open the staging folder automatically
Start-Process explorer.exe $stagingDir

Start-Sleep -Seconds 5