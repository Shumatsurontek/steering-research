---
description: Sync PLAN.md, README.md, and article/main.tex with latest experimental results
---

Update all documentation files with the latest results from `results/` directory.

## Steps

1. **Scan results/**: list all JSON result files and their modification dates to identify what's new
2. **Read current docs**: PLAN.md, README.md, article/main.tex
3. **PLAN.md**: Update phase status (✅/🔄/⬜), add new metrics tables, update sub-phase checklists
4. **README.md**: Update experiment tables, key findings boxes, future work checklist, experiment count badge
5. **article/main.tex**: Update tables, add new paragraphs for new experiments, update abstract/conclusion if needed
6. **Show diff summary**: list what was changed in each file

## Rules
- Keep PLAN.md in French
- Keep README.md in English
- Keep article/main.tex in English (academic style)
- Don't invent results — only use data from actual result files
- Don't change the document structure — only update content within existing sections
- When adding a new experiment, follow the numbering convention in each file
- Update the experiment count badge in README.md if adding new experiments

## Important
- Read all three files before making changes
- Read relevant result JSON files to get exact numbers
- Don't commit — user manages git themselves
