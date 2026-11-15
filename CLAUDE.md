# CLAUDE.md

This repository is for the IM2C 2026 fall-foliage problem: forecasting Maine autumn foliage from MODIS reflectance,
weather, and forest covariates. 

This is a research codebase. The primary objective is scientific correctness and reproducibility, not maximum code cleverness.
Avoid introducing unnecessary abstractions, factories, dependency injection, metaclasses, or complicated design patterns.
Small duplicated code is preferable to complicated generic code.

---

# User background

I am an applied mathematician with 2-3 years of scientific computing and machine learning experience. 
I prioritize precision and formality.
I do not prioritize beautiful code. I like conciseness.

I like writing that is readable without me having to strain my eyes and brain.
I do not like the use of fancy syntax in writing, or symbols like "->". Commas and periods suffice.

---

# High-level information

- I am coding and running everything on Duke Compute Cluster.
- Assume future readers are applied mathematicians or environmental scientists, not software engineers.
- Read `README.md` and `data/README.md` for an overview of the project and the data it uses.

---

# High-level priorities

When making changes:

1. First priority is scientific correctness.
2. Second priority is simple implementations over clever abstractions.
3. Third priority is reproducibility.
4. Fourth priority is optimal readability for researchers.
5. Fifth priority is memory efficiency and speed.

---

# High-level guidelines

---

## Writing guidelines

Whenever you are writing something user-facing, i.e. a README or a markdown to log/report something,
avoid the playful, fancy jargon and syntax LLMs are optimized for, if you can help it.

---

## "Good Science" guidelines

I want to emphasize again: scientific correctness is PARAMOUNT. Follow these rules, always:
- No shortcuts. If you even get an urge to take a shortcut, stop immediately and report it.
- No reporting results without an associated, verifiable script.
- No claiming a result has numerical evidence without actually running code, i.e. no guesses based on rough scans.
- No skipping evidence or failing to report evidence which does not line up with your hypothesis or my hypothesis.
- No assumptions made without verifying with me and with the references.

---

## Python guidelines

Use

- type hints for parameters in function declaration, only when the parameter is a Python primitive or `np.ndarray`
- os.path functions instead of pathlib
- built in logging structure instead of print
- existing project functions and configs (`configs/`) whenever possible and convenient
- simple in-line comments which match the commenting style present in existing scripts in `src/`
- simple, 1-2 line docstrings for functions

Avoid

- defining new dataclasses UNLESS specified in the task
- writing long docstrings UNLESS specified in the task
- deeply nested conditionals
- helper function overload; fold single-use helpers into parent functions
- wildcard imports
- hidden side effects

---

## Scientific computing guidelines

Never:

- silently change array shapes
- silently change dtypes
- silently change units or scale factors
- silently change indexing conventions
- silently change coordinate reference systems
- silently change how missing or fill values are handled

When changing numerical code, explicitly verify:

- array dimensions and dimension order
- dtype consistency
- units and scale factors
- coordinate reference systems and grid alignment
- treatment of fill, nodata, and masked values

Whenever possible, add assertions for expected array shapes.

---

## Array and dataframe guidelines

Prefer

- vectorized NumPy, xarray, and pandas operations
- broadcasting
- named dimensions in xarray over positional indexing
- documentation of expected array shapes via inline comments

Avoid

- unnecessary Python loops, in particular loops over pixels or grid cells
- unnecessary copies of large arrays
- unnecessary conversions between NumPy, pandas, and xarray

Avoid in-place operations unless clearly beneficial.

---

## Reproducibility guidelines

Do not change:

- random seeds
- train, validation, and test splits
- evaluation protocol

unless explicitly requested.

If changes affect reproducibility, explain why.
Every downloaded or manually obtained dataset must be recorded in `data/README.md`.
Create user-friendly READMEs whenever you see fit.

---

## Comment guidelines

- in-line comments should ALMOST ALWAYS be one line; take efforts to avoid two-line comments
- comments SHOULD NOT use math symbols; those look pretty when rendered but are utterly unreadable and non-sensical in plain text
- comments SHOULD NOT use variables, EVEN IF you define them, e.g. "NxN"; just write out what the thing is in words
- comments NEED NOT be fanciful sentences with fancy diction; no human should need to do a double take when reading them
- in fact, comments SHOULD NOT CONTAIN PERIODS, as they are not sentences, but comments
- following up the previous point, comments are NOT sensationalist media/headlines; treat them like sentences you would see in a textbook or a research paper: precise, concise, says exactly what it needs to say without fancy syntax or diction, without pretty words like "fail silently", without fancy punctuation
- if multiple lines is really, truly necessary to clearly communicate what is going on, please adhere to these other guidelines

Remember that comments are meant to help the coder understand code, not to show off how efficiently and beautifully you can explain the code.
Buzz words like "silently", "quietly", "loudly", etc. are NOT how coders talk to each other, and should thus not be how you write comments.

---

## SLURM guidelines

- you may run jobs if I instruct you to, but ALWAYS report when you do and ask permission
- to repeat, you MUST ask permission to run a series of jobs (this does not mean you need to stop between every job; if I approve you to run a series of related jobs, monitor them, and queue new ones based on their performance, you may do so)
- follow the conventions in the existing SLURM scripts in `src/shell_scripts/`

---

## What not to do

Do not:

- rewrite large sections without justification
- introduce unnecessary dependencies
- rename APIs purely for style
- change file organization unless it materially improves the project
- optimize away code that improves readability

NEVER WRITE/MODIFY/OUTPUT CODE until explicitly prompted by me. The default is planning, not coding.
NEVER add a new dependency without seeking explicit permission in the planning phase.
NEVER do any Git commands (stage changes, commit, push, pull) without very explicit permission.

---

## When uncertain

If multiple implementations appear reasonable:

1. preserve existing conventions
2. choose the simplest implementation and explain it VERY precisely
3. explain assumptions
4. leave TODO comments rather than guessing
5. explain the alternative implementations, also VERY precisely

Scientific correctness always takes precedence over code elegance.
