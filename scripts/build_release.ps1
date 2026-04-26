param(
    [string]$VenvPython = ".venv\\Scripts\\python.exe",
    [string]$SpecPath = "packaging\\pyinstaller\\aura.spec",
    [string]$RuntimeRoot = ".runtime",
    [string]$ReleaseName = "aura-release",
    [string]$PyInstallerVersion = "6.14.2",
    [switch]$IncludeNvidia,
    [switch]$SkipBuild,
    [switch]$SkipAssemble
)

$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param([string]$PathValue, [string]$Label)
    if (-not (Test-Path $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Invoke-RobocopySafe {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$ExtraArgs = @()
    )

    Assert-PathExists -PathValue $Source -Label "Robocopy source"
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    $args = @(
        $Source,
        $Destination,
        "/E",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    ) + $ExtraArgs

    & robocopy @args | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "Robocopy failed with exit code $code while copying '$Source' -> '$Destination'."
    }
}

function Copy-PlanPackages {
    param(
        [string]$Source,
        [string]$Destination
    )

    Assert-PathExists -PathValue $Source -Label "Plans directory"
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    Get-ChildItem -LiteralPath $Source -File -Filter "*.py" |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force
        }

    $excludedPlanDirectoryNames = @("logs", "__pycache__", ".pytest_cache")
    $planPackages = Get-ChildItem -LiteralPath $Source -Directory |
        Where-Object {
            $excludedPlanDirectoryNames -notcontains $_.Name -and
            (Test-Path (Join-Path $_.FullName "manifest.yaml"))
        }

    foreach ($planPackage in $planPackages) {
        Invoke-RobocopySafe `
            -Source $planPackage.FullName `
            -Destination (Join-Path $Destination $planPackage.Name) `
            -ExtraArgs @("/XD", "__pycache__", ".pytest_cache", "/XF", "*.pyc", "*.pyo")
    }
}

function Ensure-PyInstaller {
    param(
        [string]$PythonPath,
        [string]$Version
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonPath -c "import PyInstaller" *> $null
        $hasPyInstaller = $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($hasPyInstaller) {
        return
    }

    Write-Host "Installing PyInstaller $Version into build venv ..."
    & $PythonPath -m pip install "pyinstaller==$Version"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller $Version."
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPythonPath = Join-Path $RepoRoot $VenvPython
$SpecFilePath = Join-Path $RepoRoot $SpecPath
$RuntimeRootPath = Join-Path $RepoRoot $RuntimeRoot
$DistPath = Join-Path $RuntimeRootPath "dist"
$WorkPath = Join-Path $RuntimeRootPath "build\\pyinstaller"
$ReleaseRoot = Join-Path $RuntimeRootPath "release\\$ReleaseName"
$BuiltRuntimeDir = Join-Path $DistPath "aura"
$ReleaseRuntimeDir = Join-Path $ReleaseRoot "runtime"
$ReleasePlansDir = Join-Path $ReleaseRoot "plans"
$RunTemplate = Join-Path $RepoRoot "packaging\\templates\\run.ps1"
$ConfigTemplate = Join-Path $RepoRoot "packaging\\templates\\config.yaml"
$SourcePlansDir = Join-Path $RepoRoot "plans"
$SourceLicense = Join-Path $RepoRoot "LICENSE"
$SourceReadme = Join-Path $RepoRoot "README.md"

Assert-PathExists -PathValue $VenvPythonPath -Label "Venv python"
Assert-PathExists -PathValue $SpecFilePath -Label "PyInstaller spec"
Assert-PathExists -PathValue $RunTemplate -Label "Run script template"
Assert-PathExists -PathValue $ConfigTemplate -Label "Config template"
Assert-PathExists -PathValue $SourcePlansDir -Label "Plans directory"

$env:PYTHONNOUSERSITE = "1"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:AURA_PKG_INCLUDE_NVIDIA = if ($IncludeNvidia) { "1" } else { "0" }

Ensure-PyInstaller -PythonPath $VenvPythonPath -Version $PyInstallerVersion

if (-not $SkipBuild) {
    Write-Host "Building Aura runtime with PyInstaller ..."
    New-Item -ItemType Directory -Force -Path $DistPath | Out-Null
    New-Item -ItemType Directory -Force -Path $WorkPath | Out-Null

    & $VenvPythonPath -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $DistPath `
        --workpath $WorkPath `
        $SpecFilePath

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE."
    }
}

if (-not $SkipAssemble) {
    Assert-PathExists -PathValue $BuiltRuntimeDir -Label "Built runtime directory"

    if (Test-Path $ReleaseRoot) {
        Remove-Item -LiteralPath $ReleaseRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot "logs") | Out-Null

    Write-Host "Assembling release root ..."
    Invoke-RobocopySafe -Source $BuiltRuntimeDir -Destination $ReleaseRuntimeDir
    Copy-PlanPackages -Source $SourcePlansDir -Destination $ReleasePlansDir

    Copy-Item -LiteralPath $RunTemplate -Destination (Join-Path $ReleaseRoot "run.ps1") -Force

    $releaseConfigPath = Join-Path $ReleaseRoot "config.yaml"
    if (-not (Test-Path $releaseConfigPath)) {
        Copy-Item -LiteralPath $ConfigTemplate -Destination $releaseConfigPath -Force
    }

    if (Test-Path $SourceLicense) {
        Copy-Item -LiteralPath $SourceLicense -Destination (Join-Path $ReleaseRoot "LICENSE") -Force
    }
    if (Test-Path $SourceReadme) {
        Copy-Item -LiteralPath $SourceReadme -Destination (Join-Path $ReleaseRoot "README.md") -Force
    }

    $summaryPath = Join-Path $ReleaseRoot "BUILD-INFO.txt"
    @(
        "release_name=$ReleaseName"
        "built_at_utc=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
        "include_nvidia=$($IncludeNvidia.IsPresent)"
        "base_path_mode=release_root"
        "entrypoint=run.ps1"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "Release assembled at: $ReleaseRoot"
}
