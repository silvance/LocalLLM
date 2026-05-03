# Verify each installer in installers/ matches the SHA-256 recorded in
# bundle_stamp.json. Run automatically by install.bat before any installer
# is executed.

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$stampPath = Join-Path $bundle "bundle_stamp.json"

if (-not (Test-Path $stampPath)) {
    Write-Host "verify-installers: bundle_stamp.json missing" -ForegroundColor Yellow
    exit 0  # Not fatal — install.bat already warned about the missing stamp.
}

$stamp = Get-Content $stampPath -Raw | ConvertFrom-Json
$expected = $stamp.installers
if (-not $expected -or $expected.Count -eq 0) {
    Write-Host "verify-installers: bundle stamp lists no installers (was the bundle built before they were added?)" -ForegroundColor Yellow
    exit 0
}

$failed = 0
foreach ($entry in $expected) {
    $relName = $entry.name
    $expectedSha = $entry.sha256.ToLower()
    $abs = Join-Path $bundle "installers" | Join-Path -ChildPath $relName

    if (-not (Test-Path $abs)) {
        Write-Host "MISSING: installers/$relName" -ForegroundColor Red
        $failed += 1
        continue
    }
    $actual = (Get-FileHash $abs -Algorithm SHA256).Hash.ToLower()
    if ($actual -eq $expectedSha) {
        Write-Host "OK: installers/$relName" -ForegroundColor Green
    } else {
        Write-Host "MISMATCH: installers/$relName" -ForegroundColor Red
        Write-Host "  expected: $expectedSha" -ForegroundColor Red
        Write-Host "  actual:   $actual" -ForegroundColor Red
        $failed += 1
    }
}

if ($failed -gt 0) {
    Write-Host "verify-installers: $failed mismatches" -ForegroundColor Red
    exit 1
}
exit 0
