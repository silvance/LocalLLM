# Verify bundle file integrity against SHA256SUMS.txt.
# Run with:  powershell -ExecutionPolicy Bypass -File verify.ps1

[CmdletBinding()]
param(
    [string]$Manifest = "SHA256SUMS.txt"
)

$ErrorActionPreference = "Stop"
$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$manPath = Join-Path $bundle $Manifest

if (-not (Test-Path $manPath)) {
    Write-Host "Manifest not found: $manPath" -ForegroundColor Red
    exit 1
}

$lines = Get-Content $manPath
$ok = 0
$failed = 0
$missing = 0

foreach ($line in $lines) {
    if (-not ($line -match "^([0-9a-fA-F]{64})\s+\*?(.+)$")) {
        continue
    }
    $expected = $matches[1].ToLower()
    $relPath  = $matches[2]
    $absPath  = Join-Path $bundle $relPath

    if (-not (Test-Path $absPath)) {
        Write-Host "MISSING: $relPath" -ForegroundColor Red
        $missing += 1
        continue
    }
    $actual = (Get-FileHash $absPath -Algorithm SHA256).Hash.ToLower()
    if ($actual -eq $expected) {
        $ok += 1
    } else {
        Write-Host "MISMATCH: $relPath" -ForegroundColor Red
        $failed += 1
    }
}

$summary = "Verified: $ok ok, $failed mismatch, $missing missing"
if (($failed + $missing) -eq 0) {
    Write-Host $summary -ForegroundColor Green
    exit 0
} else {
    Write-Host $summary -ForegroundColor Red
    exit 1
}
