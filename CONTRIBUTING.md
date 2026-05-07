# Contributing

This project is being modernized from an early Python port of MATLAB research code.

## Development Principles

- Keep physics and validation traceable.
- Prefer small, reviewable changes.
- Do not introduce commercial-tool dependencies into the core package.
- Add tests before major refactors where behavior already exists.
- Document formulas, units, and references near the implementation.

## Local Setup

```powershell
conda env create -f environment.yml
conda activate drivetrain-python
pytest
```

## Issue Workflow

Use the backlog in `docs/github-project.md` to create GitHub issues.

Each issue should include:

- scope;
- files likely affected;
- validation or test expectation;
- any external reference used.
