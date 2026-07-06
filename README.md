# Aura_script

Aura_script is the upstream game scripting framework for downstream Aura game automation projects.

This repository contains the reusable runtime, task engine, shared plan packages, local runner SDK, diagnostics, and framework tests. Game-specific plans, GUI frontends, model assets, and business data live in downstream repositories.

## Repository Shape

```text
cli.py                         CLI entrypoint for framework/runtime commands
packages/aura_core             Scheduler, task runtime, manifest tooling, services
packages/aura_game             Local runner facade used by CLI, GUI, and tools
plans/aura_base                Shared runtime actions and platform adapters
plans/aura_benchmark           Device-independent framework smoke plan
requirements                   Runtime/dev/optional dependency profiles
scripts                        Framework setup and diagnostic helpers
tools                          Runtime, plan, OCR, YOLO, and debugging tools
tests                          Framework and shared runtime tests
docs                           Framework documentation and schemas
```

## Quick Commands

```powershell
python cli.py games --all
python cli.py tasks aura_benchmark
python cli.py run aura_benchmark tasks:single_sleep.yaml:single_sleep --inputs '{"seconds": 0.1}'
```

## Validation

```powershell
python -m packages.aura_core.cli.package_cli check plans/aura_base
python -m packages.aura_core.cli.package_cli validate plans/aura_base
python -m packages.aura_core.cli.package_cli check plans/aura_benchmark
python -m packages.aura_core.cli.package_cli validate plans/aura_benchmark
python tools\plan_doctor.py --plan aura_base
python tools\plan_doctor.py --plan aura_benchmark
```
