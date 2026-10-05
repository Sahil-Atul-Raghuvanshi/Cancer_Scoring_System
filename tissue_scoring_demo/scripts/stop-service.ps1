<#
.SYNOPSIS
  Stop everything serving one of the demo's ports.

.DESCRIPTION
  Called by stop.bat. Lives in its own file rather than inline in the batch
  script because the escaping needed to embed this much PowerShell in a .bat is
  where the last version quietly broke: it died partway through and left the
  port held.

  Three things have to be caught, and the port owner alone is none of them:

    1. `uvicorn --reload` is a reloader parent plus a worker child; `npm run
       dev` is npm plus vite. Killing the listener leaves its sibling running.
    2. A child can inherit the listening socket, so the port stays held by a
       process id that no longer exists. Killing "the owner" then fails
       silently and the port is still busy.
    3. On Windows two processes can end up bound to the same port, so the next
       start looks healthy while stale code answers requests.

  This kills the listeners, their whole descendant trees, and any process whose
  command line still names the service - while refusing to kill itself or its
  own ancestors, which is what truncated the previous version mid-run.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$Match
)

$ErrorActionPreference = 'SilentlyContinue'

function Get-Listeners {
    param([int]$Port)
    $ids = @()
    try {
        $ids += (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess)
    }
    catch {
        # Older systems, or a locked-down environment: fall back to netstat.
        $ids += netstat -ano |
            Select-String (':' + $Port + '\s.*LISTENING') |
            ForEach-Object { ($_ -split '\s+')[-1] }
    }
    return $ids | ForEach-Object { [int]$_ } | Sort-Object -Unique
}

$procs = @(Get-CimInstance Win32_Process)

# Never kill this script, the console hosting it, or anything it descends from.
$protected = [System.Collections.Generic.HashSet[int]]::new()
$walk = $PID
while ($walk -and $protected.Add([int]$walk)) {
    $walk = ($procs | Where-Object { $_.ProcessId -eq $walk } | Select-Object -First 1).ParentProcessId
}

$targets = [System.Collections.Generic.HashSet[int]]::new()
foreach ($id in (Get-Listeners -Port $Port)) { [void]$targets.Add($id) }

# The socket's owner may already be gone while a child still holds the handle,
# so match on command line too rather than trusting the port alone.
foreach ($p in $procs) {
    if ($p.CommandLine -and $p.CommandLine -like "*$Match*") { [void]$targets.Add([int]$p.ProcessId) }
}
# A dead reloader leaves multiprocessing spawn children holding the socket.
foreach ($p in $procs) {
    if ($p.CommandLine -and $p.CommandLine -like '*multiprocessing.spawn*') {
        $parent = $procs | Where-Object { $_.ProcessId -eq $p.ParentProcessId } | Select-Object -First 1
        if (-not $parent) { [void]$targets.Add([int]$p.ProcessId) }
    }
}

if ($targets.Count -eq 0) {
    Write-Host '       nothing was listening.'
    exit 0
}

# Expand each target to its full descendant tree.
$doomed = [System.Collections.Generic.HashSet[int]]::new()
$queue = [System.Collections.Queue]::new()
foreach ($t in $targets) { $queue.Enqueue([int]$t) }

while ($queue.Count -gt 0) {
    $id = [int]$queue.Dequeue()
    if ($protected.Contains($id)) { continue }
    if (-not $doomed.Add($id)) { continue }
    foreach ($child in ($procs | Where-Object { $_.ParentProcessId -eq $id })) {
        $queue.Enqueue([int]$child.ProcessId)
    }
}

# Children first, so a parent cannot respawn one on its way out.
foreach ($id in ($doomed | Sort-Object -Descending)) {
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    if ($proc.ProcessName -eq 'conhost') { continue }  # dies with its console
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    Write-Host ('       stopped ' + $proc.ProcessName + ' (pid ' + $id + ')')
}

Start-Sleep -Milliseconds 500

$left = @(Get-Listeners -Port $Port)
if ($left.Count -gt 0) {
    Write-Host ('       [WARN] port ' + $Port + ' still held by pid ' + ($left -join ', '))
    exit 1
}

Write-Host ('       port ' + $Port + ' is free.')
exit 0
