$lines = [IO.File]::ReadAllLines("app.py")
$startLamp = -1
$endLamp = -1
$startOutlet = -1
$endOutlet = -1

for ($i=0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match '<!-- Lamp Analytics Page -->') { $startLamp = $i }
    if ($lines[$i] -match '<!-- Energy Usage Page -->') { $endLamp = $i - 1 }
    if ($lines[$i] -match '<!-- Outlet Analytics Page -->') { $startOutlet = $i }
    if ($lines[$i] -match '<!-- Toast Notification -->') { $endOutlet = $i - 1 }
}

Write-Output "Start Lamp: $startLamp, End Lamp: $endLamp, Start Outlet: $startOutlet, End Outlet: $endOutlet"

if ($startLamp -ne -1 -and $endLamp -ne -1 -and $startOutlet -ne -1 -and $endOutlet -ne -1) {
    $outletBlock = $lines[$startOutlet..$endOutlet]
    $lines1 = $lines[0..($startOutlet-1)]
    $lines2 = $lines[($endOutlet+1)..($lines.Length-1)]
    
    # + is array concatenation in PowerShell
    $newLinesWithoutOutlet = @()
    $newLinesWithoutOutlet += $lines1
    $newLinesWithoutOutlet += $lines2
    
    $part1 = $newLinesWithoutOutlet[0..$endLamp]
    $part2 = $newLinesWithoutOutlet[($endLamp+1)..($newLinesWithoutOutlet.Length-1)]
    
    $finalLines = @()
    $finalLines += $part1
    $finalLines += $outletBlock
    $finalLines += $part2
    
    [IO.File]::WriteAllLines("app.py", $finalLines, [Text.Encoding]::UTF8)
    Write-Output "Successfully moved."
} else {
    Write-Output "Failed to find indices."
}
