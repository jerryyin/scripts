function Copy-RcFixture([string]$Source, [string]$Target) {
    [void][IO.Directory]::CreateDirectory($Target)
    if (!(Test-Path -LiteralPath (Join-Path $Target '.git'))) { Invoke-Native git @('init', '-q', $Target) }
    $records = @(& git -C $Source ls-files --stage)
    if ($LASTEXITCODE) { throw 'Cannot inventory the candidate rc_files index.' }
    $names = @($records | ForEach-Object { ($_ -split "`t", 2)[1] })
    $targetRoot = [IO.Path]::GetFullPath($Target).TrimEnd('\') + '\'
    foreach ($obsolete in @(& git -C $Target ls-files)) {
        if ($names -contains $obsolete) { continue }
        $path = [IO.Path]::GetFullPath((Join-Path $Target $obsolete))
        if (!$path.StartsWith($targetRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe fixture path.' }
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    $aliases = @()
    foreach ($record in $records) {
        $header, $name = $record -split "`t", 2
        $sourcePath = Join-Path $Source $name
        $destination = Join-Path $Target $name
        [void][IO.Directory]::CreateDirectory((Split-Path $destination -Parent))
        if ($header.StartsWith('120000 ')) {
            $item = Get-Item -LiteralPath $sourcePath -Force
            $content = if ($item.LinkType) { [string](@($item.Target)[0]) } else { [IO.File]::ReadAllText($sourcePath) }
            Save-Text $destination $content ''
            $aliases += $name
        } else { Copy-Item -LiteralPath $sourcePath -Destination $destination -Force }
    }
    Invoke-Native git @('-C', $Target, '-c', 'core.autocrlf=false', '-c', 'core.symlinks=false', 'add', '-f', '--all')
    foreach ($name in $aliases) {
        $blob = Invoke-Native git @('-C', $Target, 'hash-object', '-w', '--', $name) -Capture
        Invoke-Native git @('-C', $Target, 'update-index', '--add', '--cacheinfo', "120000,$blob,$name")
    }
}
