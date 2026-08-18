# Single GPU Setup Guide

This guide shows how to set up Deep Researcher Agent for a single-GPU research project.

## 1. Create Your Project

```bash
mkdir my_project && cd my_project
```

## 2. Write a Project Brief

Create `PROJECT_BRIEF.md` with:
- Research goal and target metrics
- Codebase location and structure
- GPU constraints
- Any specific requirements

## 3. Create Config

Copy and modify `config.yaml` from the repo root.

## 4. Launch

```bash
python -m core.nodes --project . --gpu 0
```

## 5. Monitor

Check `workspace/MEMORY_LOG.md` for the agent's progress.
Check `autoresearcher.log` for detailed logs.

## Tips

- Start with `--max-cycles 3` to verify everything works
- Keep PROJECT_BRIEF.md concise (under 3000 chars)
- Use directives to redirect the agent if it goes off track
