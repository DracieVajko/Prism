<#
.SYNOPSIS
    Remove the legacy Odysseus/WhisperApi services and their files.

.DESCRIPTION
    Before Prism there were two separate products on this machine:

      * OdysseusDiffusion  - diffusion_server.py behind NSSM, port 8101
      * whisper-api        - "C:\Program Files\Whisper Api", port 4444

    Prism now serves both on port 8101, so both are dead weight: whisper-api is
    still set to Automatic and would keep binding 4444 on every boot.

    This script stops and deletes the services, removes their files, and reports
    what it could not touch. It never deletes the Prism repo.

    Requires an elevated PowerShell. Run with -WhatIf first.

.EXAMPLE
    # Preview only
    powershell -ExecutionPolicy Bypass -File pc\scripts\remove-legacy-services.ps1 -WhatIf

    # Actually remove
    powershell -ExecutionPolicy Bypass -File pc\scripts\remove-legacy-services.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string[]]$LegacyServices = @("OdysseusDiffusion", "whisper-api"),
    [string[]]$LegacyDirs = @("C:\Program Files\Whisper Api"),
    [string]$LegacyNssm = "D:\AI\DiffusionServer\nssm.exe",
    [switch]$KeepFiles
)

$ErrorActionPreference = "Continue"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

$isAdmin = Test-Admin
Write-Host "Prism legacy cleanup"
Write-Host "  elevated: $isAdmin"
Write-Host ""

if (-not $isAdmin) {
    Write-Warning "Not elevated. Service removal needs Administrator; re-run from an elevated shell."
    Write-Warning "Nothing was changed."
    return
}

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
foreach ($name in $LegacyServices) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if (-not $svc) {
        Write-Host "service $name : not present"
        continue
    }

    Write-Host "service $name : status=$($svc.Status) start=$($svc.StartType)"

    if ($PSCmdlet.ShouldProcess($name, "stop and delete service")) {
        if ($svc.Status -ne "Stopped") {
            Stop-Service -Name $name -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }

        # sc.exe is the reliable way to delete a service; Remove-Service needs
        # PowerShell 6+ and fails on services with a pending stop.
        $out = & sc.exe delete $name 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  deleted"
        } else {
            Write-Warning "  sc delete failed: $out"
        }
    }
}

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
if (-not $KeepFiles) {
    foreach ($dir in $LegacyDirs) {
        if (-not (Test-Path $dir)) {
            Write-Host "dir $dir : not present"
            continue
        }
        $size = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object Length -Sum).Sum
        $mb = if ($size) { [math]::Round($size / 1MB, 1) } else { 0 }
        Write-Host "dir $dir : $mb MB"

        if ($PSCmdlet.ShouldProcess($dir, "delete directory")) {
            try {
                Remove-Item -Path $dir -Recurse -Force -ErrorAction Stop
                Write-Host "  deleted"
            } catch {
                Write-Warning "  could not delete: $($_.Exception.Message)"
            }
        }
    }

    if (Test-Path $LegacyNssm) {
        Write-Host "file $LegacyNssm : $((Get-Item $LegacyNssm).Length) bytes"
        if ($PSCmdlet.ShouldProcess($LegacyNssm, "delete file")) {
            Remove-Item -Path $LegacyNssm -Force -ErrorAction SilentlyContinue
            Write-Host "  deleted (canonical copy stays in pc\scripts\nssm.exe)"
        }
    }
}

# ---------------------------------------------------------------------------
# Leftovers
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Remaining services matching Odysseus/whisper/Prism:"
$left = Get-CimInstance Win32_Service -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "Odysseus|whisper|Prism" } |
    Select-Object Name, State, StartMode
if ($left) {
    $left | Format-Table -AutoSize | Out-String | Write-Host
} else {
    Write-Host "  (none)"
}

Write-Host "Ports 8101 and 4444:"
foreach ($p in @(8101, 4444)) {
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Write-Host "  $p : LISTENING (PID $($conn[0].OwningProcess))"
    } else {
        Write-Host "  $p : free"
    }
}

Write-Host ""
Write-Host "Done. If 4444 is still listening, the process was started outside the"
Write-Host "service manager; reboot or stop it manually."
