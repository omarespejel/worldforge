# Introduction

WorldForge is a Python-first orchestration layer for world model workflows.

The project used to center a Rust workspace with Python bindings. That design has been retired. The current direction optimizes for the ecosystem that actually exists: provider SDKs, experiment tooling, and operational workflows are predominantly Python.

## Current focus

- make the package easy to install and modify
- keep the provider abstraction honest about maturity
- support local development with a deterministic mock provider
- preserve clean API boundaries for prediction, planning, evaluation, and verification

## Current implementation status

- fully implemented in-repo: mock provider, package API, CLI, JSON persistence, evaluation/report rendering, verification bundles
- scaffolded only: remote provider adapters
- intentionally deferred: production REST server rebuild
