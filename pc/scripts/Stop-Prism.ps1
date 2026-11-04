# Stop the Prism server (Windows).
#
# 1. Stops the Prism Windows service if it exists.
# 2. Stops a Prism started from the Startup shortcut.
# 3. Falls back to killing whatever listens on the port.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File pc\scripts\Stop-Prism.ps1
#   powershell -ExecutionPolicy Bypass -File pc\scripts\Stop-Prism.ps1 -Port 8102

[CmdletBinding()]
param(
    [int]$Port = 8101,
    [string]$ServiceName = "Prism"
)

$ErrorActionPreference = "Continue"

Write-Host "Stopping Prism..."

# 1. Windows service
$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne "Stopped") {
        Write-Host "  stopping service $ServiceName (status=$($svc.Status))"
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    } else {
        Write-Host "  service $ServiceName already stopped"
    }
}

# 2. Autostart launcher process
$runners = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "start-prism-autostart\.ps1" }
foreach ($r in $runners) {
    Write-Host "  stopping autostart runner PID $($r.ProcessId)"
    Stop-Process -Id $r.ProcessId -Force -ErrorAction SilentlyContinue
}

# 3. Anything else holding the port
$owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid_ in $owners) {
    if (-not $pid_) { continue }
    try {
        $p = Get-Process -Id $pid_ -ErrorAction Stop
        Write-Host "  port $Port held by $($p.ProcessName) (PID $pid_), stopping"
        Stop-Process -Id $pid_ -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host "  port $Port held by PID $pid_ (cannot inspect)"
    }
}

Start-Sleep -Seconds 1

$still = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($still) {
    Write-Warning "Port $Port is still listening. Run this script from an elevated shell if the process belongs to a service."
} else {
    Write-Host "Prism stopped; port $Port is free."
}
