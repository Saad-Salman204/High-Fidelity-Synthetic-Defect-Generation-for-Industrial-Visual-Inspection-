#requires -Version 5.1
<#
.SYNOPSIS
Guided local execution of the university lab workflow.
.EXAMPLE
.\run_lab.ps1 -DryRun -Action plan
#>
[CmdletBinding()]
param(
    [ValidateSet('menu', 'plan', 'configure', 'environment', 'setup', 'templates',
        'models', 'audit', 'freeze', 'baseline', 'lora', 'generate', 'review',
        'quality', 'classifiers', 'selection', 'evaluate', 'backup', 'tests')]
    [string]$Action = 'menu',
    [string]$SessionFile = 'artifacts/lab-session.json',
    [string]$Option = '',
    [switch]$DryRun,
    [switch]$NonInteractive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Root = $PSScriptRoot
$script:LogPath = $null
$script:SessionPath = $null
$script:Lock = $null
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-Event([string]$Event, $Details) {
    if ($script:LogPath) {
        $entry = @{ time = [DateTime]::UtcNow.ToString('o'); event = $Event
                    action = $script:CurrentAction; details = $Details }
        [IO.File]::AppendAllText($script:LogPath,
            (($entry | ConvertTo-Json -Depth 12 -Compress) + [Environment]::NewLine), $script:Utf8)
    }
}

function Local-Path([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or [IO.Path]::IsPathRooted($Value) -or
        $Value -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "Use a relative path inside the checkout: $Value"
    }
    $path = [IO.Path]::GetFullPath((Join-Path $script:Root $Value))
    $boundary = $script:Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $path.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Path must stay below the checkout root.'
    }
    $ancestor = $path
    while ($ancestor -and $ancestor -ne $script:Root) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Linked paths require review outside the launcher: $Value"
            }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    return $path
}

function Save-Session {
    if ($DryRun) { return }
    $parent = Split-Path -Parent $script:SessionPath
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    $temp = $script:SessionPath + '.' + [guid]::NewGuid().ToString('N')
    [IO.File]::WriteAllText($temp, ($script:Settings | ConvertTo-Json -Depth 12), $script:Utf8)
    Move-Item -LiteralPath $temp -Destination $script:SessionPath -Force
}

function File-Hash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally { $hasher.Dispose(); $stream.Dispose() }
}

function Confirm-Step([string]$Word, [string]$Message) {
    Write-Host $Message
    if ($DryRun) { Write-Host "[preview] Would require typing $Word"; return }
    if ($NonInteractive) { throw "This action requires a person to type $Word. Run the menu." }
    if ((Read-Host "Type $Word to continue") -cne $Word) { throw 'Action cancelled.' }
    Write-Event 'confirmation' @{ confirmation = $Word }
}

function Invoke-Program {
    param([string]$Program, [string[]]$Arguments,
          [int[]]$AllowedCodes = @(0), [switch]$Capture)
    $shown = (@($Program) + $Arguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ' '
    Write-Host "& $shown"
    if ($DryRun) { return }
    Write-Event 'command' @{ program = $Program; arguments = $Arguments }
    $lines = New-Object 'System.Collections.Generic.List[string]'
    $savedPreference = $ErrorActionPreference
    try {
        # PowerShell 5.1 can represent native stderr as ErrorRecord on success.
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        & $Program @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $lines.Add($line)
            Write-Host $line
            Write-Event 'output' $line
        }
        $code = $global:LASTEXITCODE
    } catch {
        throw "Command failed to start: $Program. $($_.Exception.Message)"
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    Write-Event 'exit' @{ code = $code; program = $Program }
    if ($null -eq $code -or $code -notin $AllowedCodes) {
        throw "Command failed (exit $code). Inspect the log and preserve the run outputs."
    }
    if ($code -ne 0) { Write-Host "Exit $code requires review. Read the report before proceeding." }
    if ($Capture) { return ($lines -join [Environment]::NewLine) }
}

function Invoke-Python {
    param([string[]]$Arguments, [int[]]$AllowedCodes = @(0), [switch]$Capture)
    if (-not $DryRun) {
        foreach ($approval in $script:ActiveApprovals) {
            if ((File-Hash (Local-Path $approval.path)) -ne $approval.sha256) {
                throw "Approved input changed during this action: $($approval.path)"
            }
        }
    }
    $options = @{ Program = (Local-Path $script:Settings.python); Arguments = $Arguments
                  AllowedCodes = $AllowedCodes; Capture = $Capture }
    Invoke-Program @options
}

function Get-Approval([string]$Name, [string]$File, [string]$Meaning) {
    $path = Local-Path $File
    Write-Host "Required review: $Meaning"
    if ($DryRun) {
        Write-Host "[preview] Verify the named reviewer and independently approved SHA-256 of $File"
        return "<approved-$Name-sha256>"
    }
    $digest = File-Hash $path
    $entry = $script:Settings.approvals.PSObject.Properties[$Name]
    if ($entry -and $entry.Value.path -eq $File -and $entry.Value.sha256 -eq $digest -and
        -not [string]::IsNullOrWhiteSpace($entry.Value.reviewer) -and
        -not [string]::IsNullOrWhiteSpace($entry.Value.note)) {
        Write-Event 'approval_reused' @{ name = $Name; path = $File; sha256 = $digest }
        $script:ActiveApprovals.Add($entry.Value)
        return $digest
    }
    if ($NonInteractive) { throw "Missing or stale $Name approval for $File. Use the menu." }
    Write-Host "Current file SHA-256: $digest"
    $approved = (Read-Host 'Paste the SHA-256 from the independent review').Trim().ToLowerInvariant()
    if ($approved -notmatch '^[a-f0-9]{64}$' -or $approved -ne $digest) {
        throw 'Approved hash does not match the current artifact.'
    }
    $reviewer = (Read-Host 'Who reviewed this artifact?').Trim()
    $note = (Read-Host 'Record the review conclusion, permission record or pilot evidence').Trim()
    if (-not $reviewer -or -not $note) { throw 'Reviewer and review evidence are required.' }
    $approval = [pscustomobject]@{ path = $File; sha256 = $digest; reviewer = $reviewer
        note = $note; approved_at = [DateTime]::UtcNow.ToString('o') }
    $script:Settings.approvals | Add-Member -MemberType NoteProperty -Name $Name -Value $approval -Force
    Save-Session
    Write-Event 'approval' @{ name = $Name; record = $approval }
    $script:ActiveApprovals.Add($approval)
    return $digest
}

function Get-Config([string]$File) {
    $value = Invoke-Python -Arguments @('scripts/lab_support.py', 'config', $File) -Capture
    if (-not $DryRun) { return ($value | ConvertFrom-Json) }
}

function Approve-Config([string]$File, [string]$Stage) {
    $before = $null
    if (-not $DryRun) { $before = File-Hash (Local-Path $File) }
    $config = Get-Config $File
    $digest = Get-Approval "config-$Stage" $File 'Protocol, paths, precision, seeds and budget reviewed.'
    if (-not $DryRun) {
        if ($digest -ne $before) { throw 'Config changed while it was being checked. Review it again.' }
        if ($config.status -ne 'approved') {
            throw "Review $File and set status: approved in the editor before real runs."
        }
        $manifest = $script:Settings.split_dir.TrimEnd('/') + '/manifest.csv'
        if ($config.paths.manifest -ne $manifest -or
            $config.paths.data_root -ne $script:Settings.data_root) {
            throw 'Session data/split paths differ from this config. Fix the session or reviewed config.'
        }
    }
    return $config
}

function Manifest-Approval {
    $archive = Get-Approval 'dataset-use' $script:Settings.archive 'Written terms permit the intended use of this NEU-CLS archive.'
    $manifest = $script:Settings.split_dir.TrimEnd('/') + '/manifest.csv'
    $digest = Get-Approval 'manifest' $manifest 'Peer approved groups, exclusions, actual counts and the sealed real test set.'
    Invoke-Python -Arguments @('scripts/lab_support.py', 'manifest', '--csv', $manifest, '--archive-sha256', $archive)
    return $digest
}

function Check-Base {
    if ($DryRun) {
        Write-Host '[preview] Recompute base component fingerprints and compare with the reviewed model receipt.'
        return
    }
    $entry = $script:Settings.approvals.PSObject.Properties['base-model']
    if (-not $entry) { throw 'Run the models step and review the local SD fingerprint receipt first.' }
    $null = Get-Approval 'base-model' $entry.Value.path 'Base revision, component fingerprints and terms reviewed.'
    $arguments = @('scripts/lab_support.py', 'models', '--snapshot', $script:Settings.snapshot,
        '--approved-receipt', $entry.Value.path)
    if ($script:Settings.variant) { $arguments += @('--variant', $script:Settings.variant) }
    Invoke-Python -Arguments $arguments
}

function Check-Device($Config, [string]$Stage) {
    if ($DryRun) {
        Write-Host "[preview] Check the device in $Stage configuration with check_ml_environment.py."
        return
    }
    $result = Invoke-Python -Arguments @('scripts/lab_support.py', 'device',
        '--requested', ([string]$Config.$Stage.device)) -Capture
    $device = [string]($result | ConvertFrom-Json).device
    $script:ResolvedDevice = $device
    if ($device.StartsWith('cuda')) {
        Invoke-Program -Program 'nvidia-smi' -Arguments @('--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv')
        Invoke-Python -Arguments @('scripts/check_ml_environment.py', '--device', 'cuda')
    } else { Invoke-Python -Arguments @('scripts/check_ml_environment.py', '--device', 'cpu') }
}

function Invoke-Workload([string[]]$Arguments, $Config, [string]$Stage) {
    $monitor = $null
    $started = [DateTime]::UtcNow
    try {
        if (-not $DryRun) {
            Write-Event 'workload_started' @{ stage = $Stage
                requested_device = $Config.$Stage.device; resolved_device = $script:ResolvedDevice }
        }
        if (-not $DryRun -and $script:ResolvedDevice.StartsWith('cuda')) {
            $nvidia = (Get-Command nvidia-smi -CommandType Application -ErrorAction Stop).Source
            $memoryLog = $script:LogPath + '.' + [guid]::NewGuid().ToString('N') + '-gpu.csv'
            $monitorArgs = @('--query-gpu=timestamp,name,memory.used,memory.free',
                '--format=csv', '-l', '1')
            $monitor = Start-Process -FilePath $nvidia -ArgumentList $monitorArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $memoryLog -RedirectStandardError ($memoryLog + '.stderr')
            Write-Event 'gpu_monitor' @{ path = $memoryLog; pid = $monitor.Id }
            if ($monitor.WaitForExit(200)) { throw 'GPU memory sampler stopped. Inspect its stderr log.' }
        }
        Invoke-Python -Arguments $Arguments
    } finally {
        if ($monitor) {
            $monitor.Refresh()
            if (-not $monitor.HasExited) { $monitor.Kill(); $monitor.WaitForExit() }
            $monitor.Dispose()
        }
        Write-Event 'wall_seconds' (([DateTime]::UtcNow - $started).TotalSeconds)
    }
}

function Ensure-New([string]$Value) {
    $path = Local-Path $Value
    if (-not $DryRun -and (Test-Path -LiteralPath $path)) {
        throw "Output already exists: $Value. Choose a new path or the documented resume action."
    }
}

function Choose([string]$Prompt, [string[]]$Choices) {
    if ($Option -and $Action -ne 'plan' -and -not $script:ChoiceUsed) {
        if ($Option -notin $Choices) { throw 'The requested option is not valid for this step.' }
        $script:ChoiceUsed = $true
        return $Option
    }
    if ($DryRun) { Write-Host "[preview] $Prompt ($($Choices -join ', ')); showing $($Choices[0])"; return $Choices[0] }
    if ($NonInteractive) { throw 'This step requires an interactive choice.' }
    $value = Read-Host "$Prompt ($($Choices -join ', '))"
    if ($value -notin $Choices) { throw 'Choose one of the listed values.' }
    return $value
}

function Configure-Session {
    if ($DryRun) { Write-Host '[preview] Edit launcher paths; scientific configs need separate review.'; return }
    if ($NonInteractive) { throw 'Use the interactive session editor.' }
    Write-Host 'Press Enter to keep a value. Use forward slashes for paths in the checkout.'
    foreach ($property in $script:Settings.PSObject.Properties) {
        if ($property.Name -in @('schema_version', 'approvals')) { continue }
        $value = Read-Host "$($property.Name) [$($property.Value)]"
        if ($value -eq '-') { $script:Settings.($property.Name) = '' }
        elseif ($value) { $script:Settings.($property.Name) = $value.Trim() }
    }
    Validate-Session
    Save-Session
    Write-Host "Saved $SessionFile. Config files still need separate review."
}

function Run-Stage([string]$Stage) {
    $s = $script:Settings
    $script:ChoiceUsed = $false
    $script:ResolvedDevice = 'cpu'
    $script:ActiveApprovals = New-Object 'System.Collections.Generic.List[object]'
    $script:CurrentAction = $Stage
    Write-Event 'stage_started' $Stage
    switch ($Stage) {
        'configure' { Configure-Session }
        'environment' {
            Invoke-Program -Program 'git' -Arguments @('--version')
            Invoke-Python -Arguments @('--version')
            if ($s.device -eq 'cuda') {
                Invoke-Program -Program 'nvidia-smi' -Arguments @('--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv')
            } elseif ($s.device -ne 'cpu') { throw 'Session device must be cpu or cuda.' }
            Invoke-Python -Arguments @('scripts/check_ml_environment.py', '--device', $s.device)
            Invoke-Python -Arguments @('-m', 'pip', 'freeze', '--all')
        }
        'setup' {
            if ($s.wheel_index -notin @('cpu', 'cu126', 'cu130', 'cu132')) {
                throw 'Choose a reviewed official PyTorch CPU/CUDA wheel index.'
            }
            Confirm-Step 'INSTALL' 'Install pinned packages into this session virtual environment.'
            $venv = Local-Path $s.venv_dir; $python = Local-Path $s.python
            if ($python -ne (Join-Path $venv 'Scripts/python.exe')) {
                throw 'Setup requires python to name this venv_dir/Scripts/python.exe.'
            }
            if (-not $DryRun -and (Test-Path -LiteralPath $venv) -and -not (Test-Path -LiteralPath $python)) {
                throw 'Existing environment is incomplete. Choose a fresh environment path.'
            }
            if ($DryRun -or -not (Test-Path -LiteralPath $python)) {
                Invoke-Program -Program 'py' -Arguments @('-3.11', '-m', 'venv', $venv)
            }
            $torch = 'torch==2.14.1'; $vision = 'torchvision==0.29.1'
            if ($s.wheel_index -eq 'cpu') { $torch += '+cpu'; $vision += '+cpu' }
            Invoke-Python -Arguments @('-m', 'pip', 'install', $torch, $vision, '--index-url', "https://download.pytorch.org/whl/$($s.wheel_index)")
            Invoke-Python -Arguments @('-m', 'pip', 'install', '-e', '.[dev,audit,lora]')
            Invoke-Python -Arguments @('-m', 'pip', 'check')
        }
        'templates' {
            $kind = Choose 'Create a proposed template' @('base', 'classifier-pilot', 'lora-pilot', 'generation-pilot', 'full', 'generation', 'comparison')
            $target = switch ($kind) {
                'base' { $s.base_config }; 'classifier-pilot' { $s.baseline_config }
                'lora-pilot' { $s.pilot_config }; 'generation-pilot' { $s.pilot_generation_config }
                'full' { $s.full_config }; 'generation' { $s.generation_config }
                'comparison' { $s.comparison_config }
            }
            Ensure-New $target
            $source = $s.base_config
            if ($kind -eq 'base') { $source = 'configs/baseline.yaml' }
            $template = $kind
            if ($kind -in @('full', 'generation', 'comparison')) { $template = 'base' }
            Invoke-Python -Arguments @('scripts/lab_support.py', 'prepare', '--source', $source,
                '--output', $target, '--kind', $template, '--data', $s.data_root,
                '--manifest', ($s.split_dir.TrimEnd('/') + '/manifest.csv'))
            Write-Host "Edit $target in your editor. Review it, then set status: approved."
        }
        'models' {
            $null = Get-Approval 'dataset-use' $s.archive 'NEU source, acquisition, terms and intended use reviewed.'
            $null = Get-Approval 'resnet' $s.classifier_weights 'Local ImageNet V1 source, terms and full digest reviewed.'
            $arguments = @('scripts/lab_support.py', 'models', '--snapshot', $s.snapshot)
            if ($s.variant) {
                if ($s.variant -ne 'fp16') { throw 'Variant must be empty or fp16.' }
                $arguments += @('--variant', 'fp16')
            }
            $result = Invoke-Python -Arguments $arguments -Capture
            if (-not $DryRun) {
                $receipt = 'artifacts/lab-launcher/models-' + [guid]::NewGuid().ToString('N') + '.json'
                [IO.File]::WriteAllText((Local-Path $receipt), $result, $script:Utf8)
                $null = Get-Approval 'base-model' $receipt 'SD v1.5 pinned revision, component digests and terms reviewed.'
            }
            Write-Host 'Keep the local assets and acquisition records described in IMPLEMENTATION.md.'
        }
        'audit' {
            $null = Get-Approval 'dataset-use' $s.archive 'Written NEU terms and intended use reviewed.'
            Ensure-New $s.audit
            $arguments = @('scripts/audit_dataset.py', '--root', $s.data_root, '--report', $s.audit,
                '--near-threshold', $s.near_threshold, '--json-errors')
            if ($s.inventory) { $arguments += @('--inventory', $s.inventory) }
            Invoke-Python -Arguments $arguments -AllowedCodes @(0, 1)
            Invoke-Python -Arguments @('scripts/freeze_splits.py', '--audit', $s.audit, '--audit-digest')
            Write-Host "Review every finding and create $($s.dataset_review) using IMPLEMENTATION.md step 4."
        }
        'freeze' {
            $archive = Get-Approval 'dataset-use' $s.archive 'NEU terms, archive identity and intended use reviewed.'
            $null = Get-Approval 'dataset-review' $s.dataset_review 'All exclusions, duplicate pairs, conflicts and counts reviewed.'
            $config = Get-Config $s.base_config
            $seed = '42'; $ratios = @('0.7', '0.15', '0.15')
            if (-not $DryRun) {
                $seed = [string]$config.split.seed
                $ratios = @($config.split.ratios | ForEach-Object {
                    ([double]$_).ToString([Globalization.CultureInfo]::InvariantCulture)
                })
            }
            $arguments = @('scripts/freeze_splits.py', '--root', $s.data_root, '--audit', $s.audit,
                '--review', $s.dataset_review, '--archive-sha256', $archive, '--output', $s.split_dir,
                '--seed', $seed, '--ratios') + $ratios
            if ($s.inventory) { $arguments += @('--inventory', $s.inventory) }
            Invoke-Python -Arguments $arguments
            Write-Host 'Inspect split metadata and get peer approval of manifest.csv before training.'
        }
        'baseline' {
            $manifest = Manifest-Approval
            $config = Approve-Config $s.baseline_config 'baseline'
            $null = Get-Approval 'resnet' $s.classifier_weights 'ImageNet V1 source and local weights reviewed.'
            Check-Device $config 'classification'
            $seed = '42'
            if (-not $DryRun) { $seed = [string]$config.classification.seeds[0] }
            Confirm-Step 'TRAIN' 'Run the real-only pilot with the first configured seed and real validation.'
            Invoke-Workload -Arguments @('scripts/train_real_baseline.py', $s.baseline_config,
                '--seed', $seed, '--arm', 'real_only', '--approved-manifest-sha256', $manifest,
                '--local-weights', $s.classifier_weights) -Config $config -Stage 'classification'
        }
        'lora' {
            $mode = Choose 'LoRA job' @('pilot', 'full', 'resume-pilot', 'resume-full')
            $file = $s.pilot_config
            if ($mode -in @('full', 'resume-full')) { $file = $s.full_config }
            $manifest = Manifest-Approval
            $config = Approve-Config $file 'lora'
            Check-Base
            if ($mode -in @('full', 'resume-full')) {
                $null = Get-Approval 'pilot' $s.pilot_record 'GPU memory, runtime, reload/resume and six-class sample evidence support full training.'
                $receipt = '<reviewed-model-receipt>'
                if (-not $DryRun) { $receipt = $s.approvals.'base-model'.path }
                Invoke-Python -Arguments @('scripts/lab_support.py', 'pilot', '--record', $s.pilot_record,
                    '--config', $file, '--manifest-sha256', $manifest, '--receipt', $receipt)
            }
            Check-Device $config 'diffusion'
            $arguments = @('scripts/train_lora.py', $file, '--model-dir', $s.snapshot,
                '--approved-manifest-sha256', $manifest)
            if ($s.variant) { $arguments += @('--variant', $s.variant) }
            if ($mode.StartsWith('resume-')) {
                $null = Get-Approval 'resume' ($s.resume_checkpoint.TrimEnd('/') + '/checkpoint.json') 'Trusted complete intermediate checkpoint from this exact config, code, base and environment.'
                $arguments += @('--resume', $s.resume_checkpoint)
            }
            Confirm-Step 'TRAIN' 'Use the reviewed LoRA budget. Preserve the run record and monitor GPU memory.'
            Invoke-Workload -Arguments $arguments -Config $config -Stage 'diffusion'
        }
        'generate' {
            $mode = Choose 'Generation job' @('pilot', 'full', 'resume-pilot', 'resume-full')
            $file = $s.pilot_generation_config; $output = $s.pilot_generation_output
            $adapter = $s.pilot_adapter_dir
            if ($mode -in @('full', 'resume-full')) {
                $file = $s.generation_config; $output = $s.generation_output; $adapter = $s.adapter_dir
            }
            $manifest = Manifest-Approval
            $config = Approve-Config $file 'generation'
            Check-Base
            $digest = Get-Approval 'adapter' ($adapter.TrimEnd('/') + '/checkpoint.json') 'Selected trained adapter, base revision and source manifest reviewed.'
            Check-Device $config 'generation'
            $arguments = @('scripts/generate_defects.py', $file, '--model-dir', $s.snapshot,
                '--adapter-dir', $adapter, '--approved-adapter-sha256', $digest,
                '--approved-manifest-sha256', $manifest, '--output', $output)
            if ($mode.StartsWith('resume-')) {
                Confirm-Step 'RESUME' 'Confirm the previous generator stopped. Keep the same plan and batches.'
                $arguments += '--resume'
            } else { Ensure-New $output }
            Confirm-Step 'GENERATE' 'Create pending candidates for the reviewed batch plan.'
            Invoke-Workload -Arguments $arguments -Config $config -Stage 'generation'
        }
        'review' {
            $mode = Choose 'Review stage' @('audit', 'second-review', 'export')
            $manifest = Manifest-Approval
            $output = $s.review_audit_output
            $arguments = @('scripts/review_synthetic.py', $mode, '--generation', $s.generation_output,
                '--manifest', ($s.split_dir.TrimEnd('/') + '/manifest.csv'),
                '--real-audit', $s.audit, '--approved-manifest-sha256', $manifest,
                '--threshold', $s.near_threshold)
            if ($mode -eq 'second-review') {
                $null = Get-Approval 'primary' $s.primary_review 'Every candidate has an explicit decision, reviewer, timestamp and required reason.'
                $arguments += @('--primary', $s.primary_review); $output = $s.second_review_output
            }
            if ($mode -eq 'export') {
                $null = Get-Approval 'primary' $s.primary_review 'Primary decisions complete and reviewed.'
                $null = Get-Approval 'secondary' $s.secondary_review 'Different reviewer completed sampled/disputed cases and resolved disagreements.'
                $arguments += @('--primary', $s.primary_review, '--secondary', $s.secondary_review)
                $output = $s.final_review_output
            }
            Ensure-New $output
            Invoke-Python -Arguments ($arguments + @('--output', $output))
            Write-Host "Read reports in $output. Enter human decisions in the JSON files."
        }
        'quality' {
            $mode = Choose 'Quality job' @('visual', 'kid', 'setup-quality')
            if ($mode -eq 'setup-quality') {
                Confirm-Step 'INSTALL' 'Install the pinned optional feature-extraction package.'
                Invoke-Python -Arguments @('-m', 'pip', 'install', '-e', '.[quality]')
                Invoke-Python -Arguments @('-m', 'pip', 'check')
                break
            }
            $manifest = Manifest-Approval
            $null = Get-Approval 'synthetic' ($s.final_review_output.TrimEnd('/') + '/approved.jsonl') 'Final human export, class counts, provenance and quota reviewed.'
            $output = $s.quality_output
            if ($mode -eq 'kid') { $output = $s.kid_output }
            Ensure-New $output
            $arguments = @('scripts/assess_synthetic_quality.py', '--generation', $s.generation_output,
                '--real-root', $s.data_root, '--manifest', ($s.split_dir.TrimEnd('/') + '/manifest.csv'),
                '--real-audit', $s.audit, '--review', $s.final_review_output,
                '--approved-manifest-sha256', $manifest, '--output', $output)
            if ($mode -eq 'kid') {
                $digest = Get-Approval 'inception' $s.inception_weights 'Inception feature weight source, terms and checksum reviewed.'
                $arguments += @('--weights', $s.inception_weights, '--approved-weights-sha256', $digest)
            }
            Invoke-Python -Arguments $arguments
        }
        'classifiers' {
            $manifest = Manifest-Approval
            $config = Approve-Config $s.comparison_config 'comparison'
            $null = Get-Approval 'resnet' $s.classifier_weights 'Same ImageNet initialization across all arms.'
            $synthetic = $s.final_review_output.TrimEnd('/') + '/approved.jsonl'
            $digest = Get-Approval 'synthetic' $synthetic 'Human-approved export, labels, class counts and quota accepted.'
            Invoke-Python -Arguments @('scripts/lab_support.py', 'quota', '--config', $s.comparison_config, '--review', $s.final_review_output)
            Check-Device $config 'classification'
            $arm = Choose 'Train all arms or retry one failure' @('all', 'real_only', 'synthetic_only', 'real_plus_synthetic')
            $seeds = @('42', '43', '44')
            if (-not $DryRun) {
                $seeds = @($config.classification.seeds | ForEach-Object { [string]$_ })
                if ($arm -ne 'all') { $seeds = @(Choose 'Configured retry seed' $seeds) }
            }
            Confirm-Step 'TRAIN' 'Freeze the common protocol. Each attempt gets an independent record.'
            foreach ($seed in $seeds) {
                Invoke-Workload -Arguments @('scripts/train_real_baseline.py', $s.comparison_config,
                    '--seed', $seed, '--arm', $arm, '--approved-manifest-sha256', $manifest,
                    '--local-weights', $s.classifier_weights, '--synthetic-root', $s.generation_output,
                    '--synthetic-manifest', $synthetic, '--approved-synthetic-manifest-sha256', $digest) -Config $config -Stage 'classification'
            }
        }
        'selection' {
            $manifest = Manifest-Approval
            $null = Approve-Config $s.comparison_config 'comparison'
            $arguments = @('scripts/lab_support.py', 'selection', '--config', $s.comparison_config,
                '--manifest-sha256', $manifest, '--output', $s.selection)
            if ($DryRun) { Write-Host '[preview] Add human-selected run.json paths using --record.' }
            else {
                if ($NonInteractive) { throw 'Run choices must be entered interactively.' }
                Write-Host 'List one record per configured arm/seed, including failed attempts. Keep the full inventory.'
                do {
                    $record = Read-Host 'Relative run.json path, or Enter when done'
                    if ($record) { $null = Local-Path $record; $arguments += @('--record', $record) }
                } while ($record)
            }
            Ensure-New $s.selection
            Confirm-Step 'SEAL' 'Choose from protocol and validation evidence before viewing test results.'
            Invoke-Python -Arguments $arguments
        }
        'evaluate' {
            $manifest = Manifest-Approval
            $config = Approve-Config $s.comparison_config 'comparison'
            $digest = Get-Approval 'selection' $s.selection 'Peer reviewed run choices and hashes before test metrics were opened.'
            Check-Device $config 'evaluation'
            Ensure-New $s.evaluation_output
            Confirm-Step 'TEST' 'Evaluate the sealed choices once. Report all pairs, failures and missing runs.'
            Invoke-Python -Arguments @('scripts/evaluate_classifier.py', $s.comparison_config,
                '--selection', $s.selection, '--approved-selection-sha256', $digest,
                '--approved-manifest-sha256', $manifest, '--output', $s.evaluation_output) -AllowedCodes @(0, 1)
        }
        'backup' {
            $mode = Choose 'Backup content' @('code-and-artifacts', 'include-data-and-models')
            $destination = $s.backup_destination
            if (-not $destination) {
                if ($DryRun) { $destination = '<fresh-approved-external-directory>' }
                elseif ($NonInteractive) { throw 'Set backup_destination in the session editor.' }
                else { $destination = Read-Host 'Fresh directory on approved team/external storage' }
            }
            Confirm-Step 'BACKUP' 'Confirm all jobs stopped and the destination permits storing these assets.'
            $arguments = @('scripts/lab_support.py', 'backup', '--destination', $destination)
            if ($mode -eq 'include-data-and-models') { $arguments += '--include-assets' }
            Invoke-Python -Arguments $arguments
        }
        'tests' {
            Confirm-Step 'CHECK' 'Run CPU fixture checks with local examples and no pretrained downloads.'
            $old = $env:CUDA_VISIBLE_DEVICES; $threads = $env:OMP_NUM_THREADS
            try {
                if (-not $DryRun) { $env:CUDA_VISIBLE_DEVICES = '-1'; $env:OMP_NUM_THREADS = '1' }
                Invoke-Python -Arguments @('scripts/check_foundation.py')
                Invoke-Python -Arguments @('-m', 'ruff', 'check', '.')
                Invoke-Python -Arguments @('-m', 'ruff', 'format', '--check', '.')
                Invoke-Python -Arguments @('-m', 'pip', 'check')
                Invoke-Python -Arguments @('-m', 'pytest', '-q', '--strict-config', '--strict-markers')
            } finally {
                if (-not $DryRun) { $env:CUDA_VISIBLE_DEVICES = $old; $env:OMP_NUM_THREADS = $threads }
            }
        }
        default { throw "Unknown stage: $Stage" }
    }
    Write-Event 'stage_finished' $Stage
}

function Validate-Session {
    $s = $script:Settings
    if ($s.device -notin @('cpu', 'cuda') -or
        $s.wheel_index -notin @('cpu', 'cu126', 'cu130', 'cu132') -or
        $s.variant -notin @('', 'fp16')) {
        throw 'Invalid session device, wheel index or weight-file variant.'
    }
    $threshold = 0
    if (-not [int]::TryParse([string]$s.near_threshold, [ref]$threshold) -or
        $threshold -lt 0 -or $threshold -gt 64) { throw 'near_threshold must be from 0 to 64.' }
    foreach ($property in $s.PSObject.Properties) {
        if ($property.Name -notin @('schema_version', 'device', 'wheel_index', 'variant',
            'near_threshold', 'approvals', 'backup_destination') -and $property.Value) {
            $null = Local-Path ([string]$property.Value)
        }
    }
}

$defaults = [pscustomobject]@{
    schema_version = 1; python = '.venv/Scripts/python.exe'; venv_dir = '.venv'
    device = 'cpu'; wheel_index = 'cpu'; archive = 'data/local/raw/NEU-CLS.rar'
    data_root = 'data/local/raw/NEU-CLS'; audit = 'artifacts/neu-audit.json'
    inventory = ''; dataset_review = 'artifacts/neu-review.json'; split_dir = 'artifacts/neu-split-v1'
    base_config = 'artifacts/lab-base.yaml'; baseline_config = 'artifacts/lab-real-pilot.yaml'
    pilot_config = 'artifacts/lab-lora-pilot.yaml'; full_config = 'artifacts/lab-lora-full.yaml'
    pilot_generation_config = 'artifacts/lab-generation-pilot.yaml'
    generation_config = 'artifacts/lab-generation.yaml'; comparison_config = 'artifacts/lab-comparison.yaml'
    snapshot = 'models/sd-v1-5'; variant = ''; classifier_weights = 'models/resnet18-f37072fd.pth'
    inception_weights = 'models/weights-inception-2015-12-05-6726825d.pth'
    pilot_record = 'artifacts/diffusion/SELECT-PILOT-RUN/run.json'
    pilot_adapter_dir = 'artifacts/diffusion/SELECT-PILOT-RUN/step-000100'
    adapter_dir = 'artifacts/diffusion/SELECT-FULL-RUN/step-001000'
    resume_checkpoint = 'artifacts/diffusion/SELECT-RUN/step-000025'
    pilot_generation_output = 'artifacts/generation/pilot'; generation_output = 'artifacts/generation/neu-v1'
    near_threshold = '5'; review_audit_output = 'artifacts/review/first'
    primary_review = 'artifacts/review/first/primary.json'; second_review_output = 'artifacts/review/second'
    secondary_review = 'artifacts/review/second/secondary.json'; final_review_output = 'artifacts/review/final'
    quality_output = 'artifacts/quality/visual-v1'; kid_output = 'artifacts/quality/kid-v1'
    selection = 'artifacts/selection.json'; evaluation_output = 'artifacts/evaluation/frozen-comparison'
    backup_destination = ''; approvals = [pscustomobject]@{}
}

$stages = @('configure', 'environment', 'setup', 'templates', 'models', 'audit', 'freeze',
    'baseline', 'lora', 'generate', 'review', 'quality', 'classifiers', 'selection',
    'evaluate', 'backup', 'tests')
$oldLocation = Get-Location
$oldHub = $env:HF_HUB_OFFLINE; $oldTransformers = $env:TRANSFORMERS_OFFLINE
$exitCode = 0
try {
    Set-Location -LiteralPath $script:Root
    $script:CurrentAction = $Action
    $script:SessionPath = Local-Path $SessionFile
    $script:Settings = $defaults
    if (Test-Path -LiteralPath $script:SessionPath) {
        $saved = Get-Content -Raw -LiteralPath $script:SessionPath | ConvertFrom-Json
        if ($saved.schema_version -ne 1) { throw 'Unsupported session schema.' }
        foreach ($property in $saved.PSObject.Properties) {
            if (-not $defaults.PSObject.Properties[$property.Name]) { throw "Unknown session field: $($property.Name)" }
            $script:Settings.($property.Name) = $property.Value
        }
    }
    Validate-Session
    if ($Action -eq 'plan') {
        if (-not $DryRun) { throw 'Use -DryRun -Action plan to preview the entire menu.' }
        foreach ($stage in $stages) { Run-Stage $stage }
    } else {
        if (-not $DryRun) {
            Invoke-Program -Program 'git' -Arguments @('check-ignore', '-q', '--', $SessionFile)
            $logRoot = Local-Path 'artifacts/lab-launcher'
            [IO.Directory]::CreateDirectory($logRoot) | Out-Null
            $lockRoot = Local-Path '.cache/lab-launcher-locks'
            [IO.Directory]::CreateDirectory($lockRoot) | Out-Null
            $hasher = [Security.Cryptography.SHA256]::Create()
            try {
                $lockIdentity = $script:SessionPath
                if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) {
                    $lockIdentity = $lockIdentity.ToUpperInvariant()
                }
                $lockName = [BitConverter]::ToString($hasher.ComputeHash($script:Utf8.GetBytes($lockIdentity))).Replace('-', '')
            } finally { $hasher.Dispose() }
            $script:Lock = [IO.File]::Open((Join-Path $lockRoot "$lockName.lock"),
                [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
            $script:LogPath = Join-Path $logRoot ((Get-Date -Format 'yyyyMMddTHHmmss') + '-' + [guid]::NewGuid().ToString('N') + '.jsonl')
            $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
            Write-Host "Session: $SessionFile"
            Write-Host "Log: $script:LogPath"
            Write-Event 'session_started' @{ session = $SessionFile
                commit = (& git rev-parse HEAD); dry_run = $false }
        }
        if ($Action -ne 'menu') { Run-Stage $Action }
        else {
            if ($NonInteractive) { throw 'Choose -Action for noninteractive use.' }
            while ($true) {
                Write-Host 'Lab menu - choose a step, or 0 to exit. See LAB_QUICKSTART.md for the order.'
                for ($index = 0; $index -lt $stages.Count; $index++) {
                    Write-Host ("{0,2}  {1}" -f ($index + 1), $stages[$index])
                }
                $choice = Read-Host 'Step number'
                if ($choice -eq '0') { break }
                $number = 0
                if (-not [int]::TryParse($choice, [ref]$number) -or $number -lt 1 -or $number -gt $stages.Count) {
                    Write-Host 'Enter a listed number.'; continue
                }
                try { Run-Stage $stages[$number - 1] }
                catch {
                    Write-Host "STOP: $($_.Exception.Message)"
                    Write-Event 'error' $_.Exception.Message
                    Write-Host 'Resolve this failure before dependent work.'
                }
            }
        }
    }
} catch {
    [Console]::Error.WriteLine("STOP: " + $_.Exception.Message)
    Write-Event 'error' $_.Exception.Message
    $exitCode = 2
} finally {
    if ($script:Lock) { $script:Lock.Dispose() }
    $env:HF_HUB_OFFLINE = $oldHub; $env:TRANSFORMERS_OFFLINE = $oldTransformers
    Set-Location -LiteralPath $oldLocation
}
exit $exitCode
