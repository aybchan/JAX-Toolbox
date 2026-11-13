# Collect only wheel-builder Python runfile metadata; never dump the environment.
param(
    [string]$BuildRoot = 'C:/b',
    [string]$OutputPath = 'C:/b/diagnostics/python-runfiles.json'
)
$ErrorActionPreference = 'Stop'
$logicalPaths = @('_main/jaxlib/tools/platform_tags.py', '_main/jaxlib/tools/build_utils.py')
$report = [ordered]@{ build_root = $BuildRoot; trees = @(); truncated = $false; errors = @() }

function Get-PathMetadata {
    param([string]$Path)
    $record = [ordered]@{
        path = $Path; leaf_exists = $false; container_exists = $false
        item_found = $false; attributes = $null; link_type = $null; targets = @(); errors = @()
    }
    try {
        $record.leaf_exists = Test-Path -LiteralPath $Path -PathType Leaf
        $record.container_exists = Test-Path -LiteralPath $Path -PathType Container
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        $record.item_found = $true
        $record.attributes = [string]$item.Attributes
        if ($item.PSObject.Properties['LinkType']) { $record.link_type = $item.LinkType }
        if ($item.PSObject.Properties['Target']) { $record.targets = @($item.Target | Select-Object -First 4) }
    } catch {
        $record.errors += $_.Exception.Message
    }
    return $record
}

function Get-ManifestMetadata {
    param([string]$Path)
    $record = [ordered]@{ path = $Path; present = $false; matches = @(); parse_errors = 0; note = $null }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $record }
    $record.present = $true
    try {
        if ((Get-Item -LiteralPath $Path -Force).Length -gt 8MB) {
            $record.note = 'Manifest exceeds the 8 MiB inspection limit'
            return $record
        }
        $lineNumber = 0
        foreach ($line in [IO.File]::ReadLines($Path)) {
            $lineNumber++
            if ($lineNumber -gt 100000) { $record.note = 'Manifest line limit reached'; break }
            if (-not $line -or $line.Length -gt 65536) { $record.parse_errors++; continue }
            $escaped = $line.StartsWith(' ')
            $content = if ($escaped) { $line.Substring(1) } else { $line }
            $separator = $content.IndexOf(' ')
            if ($separator -lt 1) { $record.parse_errors++; continue }
            $key = $content.Substring(0, $separator)
            $target = $content.Substring($separator + 1)
            if ($escaped) {
                # Bazel escapes spaces only in keys, and newlines/backslashes in both fields.
                $key = $key.Replace('\s', ' ').Replace('\n', "`n").Replace('\b', '\')
                $target = $target.Replace('\n', "`n").Replace('\b', '\')
            }
            foreach ($logical in $logicalPaths) {
                if ($key -ne $logical -and -not $logical.StartsWith($key + '/', [StringComparison]::Ordinal)) { continue }
                if ($record.matches.Count -ge 8) { $record.note = 'Matching-entry limit reached'; return $record }
                $entry = [ordered]@{
                    line_number = $lineNumber; raw_line = $line.Substring(0, [Math]::Min($line.Length, 4096))
                    raw_line_truncated = $line.Length -gt 4096; logical_path = $logical
                    manifest_key = $key; target = $target; resolved_target = $null; resolution_note = $null
                }
                if (-not $target) {
                    $entry.resolution_note = 'Empty target was not resolved'
                } elseif ($target.Length -gt 32768 -or $target -match '[\x00-\x1f]') {
                    $entry.resolution_note = 'Invalid target path was not resolved'
                } elseif (-not [IO.Path]::IsPathFullyQualified($target)) {
                    $entry.resolution_note = 'Relative target was not resolved'
                } else {
                    $resolved = $target
                    if ($key -ne $logical) { $resolved = Join-Path $target $logical.Substring($key.Length + 1) }
                    $entry.resolved_target = Get-PathMetadata $resolved
                }
                $record.matches += $entry
            }
        }
    } catch {
        $record.note = $_.Exception.Message
    }
    return $record
}

try {
    if (Test-Path -LiteralPath $BuildRoot -PathType Container) {
        $bases = @(Get-ChildItem -LiteralPath $BuildRoot -Directory -Force | Select-Object -First 65)
        if ($bases.Count -gt 64) { $report.truncated = $true }
        :scan foreach ($base in ($bases | Select-Object -First 64)) {
            $bazelOut = Join-Path $base.FullName 'execroot/_main/bazel-out'
            if (-not (Test-Path -LiteralPath $bazelOut -PathType Container)) { continue }
            $configs = @(Get-ChildItem -LiteralPath $bazelOut -Directory -Force | Select-Object -First 65)
            if ($configs.Count -gt 64) { $report.truncated = $true }
            foreach ($config in ($configs | Select-Object -First 64)) {
                $tools = Join-Path $config.FullName 'bin/jaxlib/tools'
                if (-not (Test-Path -LiteralPath $tools -PathType Container)) { continue }
                foreach ($tree in (Get-ChildItem -LiteralPath $tools -Directory -Filter '*_wheel_tool.exe.runfiles' -Force)) {
                    if ($report.trees.Count -ge 32) { $report.truncated = $true; break scan }
                    $entry = [ordered]@{ path = $tree.FullName; runfiles = @(); manifests = @() }
                    foreach ($logical in $logicalPaths) {
                        $path = Join-Path $tree.FullName $logical
                        $item = Get-PathMetadata $path
                        $targets = @()
                        foreach ($target in $item.targets) {
                            if (-not $target -or $target.Length -gt 32768 -or $target -match '[\x00-\x1f]') { continue }
                            $resolved = if ([IO.Path]::IsPathFullyQualified($target)) { $target } else {
                                Join-Path (Split-Path -Parent $path) $target
                            }
                            $targets += Get-PathMetadata $resolved
                        }
                        $entry.runfiles += [ordered]@{ logical_path = $logical; item = $item; link_target_items = $targets }
                    }
                    foreach ($manifest in @((Join-Path $tree.FullName 'MANIFEST'), ($tree.FullName + '_manifest'))) {
                        $entry.manifests += Get-ManifestMetadata $manifest
                    }
                    $report.trees += $entry
                }
            }
        }
    } else {
        $report.errors += 'Build root is absent; no runfiles were inspected'
    }
} catch {
    $report.errors += $_.Exception.Message
}

try {
    $parent = Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath))
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputPath -Encoding utf8
    Write-Output "Python runfiles diagnostics: $($report.trees.Count) wheel tool trees; truncated=$($report.truncated); $OutputPath"
} catch {
    Write-Output "Python runfiles diagnostics could not be written: $($_.Exception.Message)"
}
