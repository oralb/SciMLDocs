# Learn SciML by growing a virtual plant

Open **[sciml_self_study.jl](sciml_self_study.jl)** in Pluto. This is a personal learning
notebook, with explanations, interactive controls, ten checked exercises, revealable
answers, and a learning log. Basic algebra and the idea of a derivative are enough
to start; an opening section explains the Julia syntax used in the examples.

All observations are synthetic. The plant-growth models are teaching examples.

## Start Pluto

The tested setup is **Julia 1.13.1 and Pluto 1.1.0**. Install Julia from
[julialang.org](https://julialang.org/downloads/) if needed. From the repository root,
install Pluto once into a separate named environment:

```sh
julia --project=@pluto -e 'using Pkg; Pkg.add(name="Pluto", version="1.1.0")'
```

Then launch the notebook:

```sh
julia --project=@pluto -e 'using Pluto; Pluto.run(notebook="notebooks/sciml_self_study.jl")'
```

If you already use Pluto, open it and select `notebooks/sciml_self_study.jl` from its
home screen. You can also download just this `.jl` file: the complete Project and
Manifest are embedded, so companion files are not required to run it. Initial package
downloads need an internet connection and compilation can take several minutes.
The notebook environment is independent of the repository's documentation environment.

Click a code cell and press **Shift+Enter** after editing. Sliders rerun the cells
that depend on them. Replace `missing` in each exercise cell with your own Julia
expression, then open the corresponding answer only when ready. Keep a copy of the
original notebook if you want to reset your work later.

## Suggested study sequence

| Session | Sections | Main outcome |
|:--|:--|:--|
| 1 | 0–2 | Julia essentials, problem/algorithm/solution, logistic growth |
| 2 | 3–4 | Analytical checks, tolerances, vector states, resource balance, events |
| 3 | 5–6 | Root finding, stability, optimization, automatic differentiation |
| 4 | 7 | Positive parameter estimation, synthetic noise, time holdout, residuals |
| 5 | 8–9 | Sensitivity checks, identifiability, learning an unknown growth term |
| 6 | 10–11 | Mini-project, learning log, troubleshooting, official references |

Allow roughly 3–5 hours including exercises; split it into short sessions.
Predict each result before running it. Change one parameter at a time and record why
the result changed. The fitting data and learned-correction example are independent
of the exploratory growth sliders, so changing the basic simulation does not silently
change the inverse problem.

The learned correction is a constrained quadratic/cubic basis model. It teaches the
differentiable-simulation loop before neural networks and links to the official UDE
examples for further study. ModelingToolkit, stiff systems, SDEs, PDEs, ensembles,
and large-scale adjoints are continuation topics, rather than additional required packages.

## Verify the calculations

From any directory:

```sh
julia /absolute/path/to/SciMLDocs/notebooks/verify.jl
```

[verify.jl](verify.jl) extracts the notebook's embedded environment to a temporary
directory, instantiates it, executes the notebook with default widget values, and
checks analytical solutions, slider-range extremes, a conserved resource balance,
event timing, equilibrium roots, optimization, parameter recovery from two starts,
training/holdout separation, derivatives, and the learned correction. Unanswered
exercises remain `missing` and do not cause the verification script to fail.

The checks support the numerical teaching examples. They do not establish the models'
adequacy for real plants or validate forecasts on independent experiments.

## Read without running Julia

In Pluto, use **Export → Static HTML** to save a readable copy with computed plots.
A static export preserves the displayed results; sliders and Julia cell edits require
reopening the `.jl` notebook in a running Pluto session.

## Source trail

The learning sequence follows the repository's
[getting-started documentation](../docs/src/getting_started/getting_started.md).
The notebook contains direct links to official Julia, Pluto, OrdinaryDiffEq,
NonlinearSolve, Optimization, SciMLSensitivity, and ModelingToolkit documentation.
Its examples and exercises are tailored for personal study rather than copied from
the documentation site's build pipeline.
