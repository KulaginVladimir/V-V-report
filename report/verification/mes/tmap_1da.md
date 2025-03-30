---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Effective diffusivity regime

```{tags} 1D, MES, transient, trapping
```

This verification case consists of a slab of depth $l = 1 \times 10^{-3} \ \mathrm{m}$ with one trap under the effective diffusivity regime.

A trapping parameter $\zeta$ is defined by

$$
    \zeta = \frac{\lambda ^ 2 \nu}{D_0 \rho} \exp \left(\frac{E_k - E_p}{k_B T}\right) + \frac{c_m}{\rho}
$$

where

$\lambda \ \mathrm{(m)}$ is the lattice parameter, \
$\nu \ (\mathrm{s}^{-1})$, the Debye frequency \
$\rho$ is the trapping site fraction, \
$c_m (\text{atom} \ \mathrm{m}^{-3})$ is the mobile atom concentration,

and the effective diffusivity $\mathrm{D_\text{eff}}$ is defined by

$$
    D_\text{eff} = \frac{D}{1 + \frac{1}{\zeta}}
$$

Then with a breakthrough time $\tau = \frac{l^2}{2\pi^2 D_\text{eff}}$, the exact solution for flux is

$$
    J = \frac{}{} \left[ 1 + 2\sum_{m=1}^\infty (-1)^m \exp \left( -m^2 \frac{t}{\tau} \right) \right]
$$

This analytical solution was obtained from TMAP7's V&V report {cite}`ambrosek_verification_2008`, case Val-1da.

+++

## FESTIM Code

```{code-cell}
:tags: [hide-input, hide-output]

import festim as F
import numpy as np

# Define input parameters
D_0 = 1.9e-7
N_A = 6.0221408e23
rho_W = 6.3382e28
trap_density = 1e-3 * rho_W
E_D = 0.2
k_0 = 1.58e7 / N_A
p_0 = 1e13
E_p = 1
T = 1000
D = D_0 * np.exp(-E_D / (F.k_B * T))
S = 2.9e-5 * np.exp(-1 / (F.k_B * T))
sample_depth = 1e-3

c_m = (1e5) ** 0.5 * S * 1.0525e5

# Create the FESTIM model
my_model = F.HydrogenTransportProblem()

vertices = np.linspace(0, sample_depth, num=100)
my_model.mesh = F.Mesh1D(vertices)

material = F.Material(D_0=D_0, E_D=E_D)
volume = F.VolumeSubdomain1D(id=1, material=material, borders=[0, sample_depth])
left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
right_boundary = F.SurfaceSubdomain1D(id=2, x=sample_depth)

my_model.subdomains = [left_boundary, volume, right_boundary]

mobile_H = F.Species("H")
trapped_H = F.Species("trapped_H", mobile=False)
empty_trap = F.ImplicitSpecies(n=trap_density, others=[trapped_H])
my_model.species = [mobile_H, trapped_H]

trapping_reaction = F.Reaction(
    reactant=[mobile_H, empty_trap],
    product=[trapped_H],
    k_0=k_0,
    E_k=E_D,
    p_0=p_0,
    E_p=E_p,
    volume=volume,
)
my_model.reactions = [trapping_reaction]

my_model.temperature = T

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=left_boundary, value=c_m * N_A, species=mobile_H),
    F.DirichletBC(subdomain=right_boundary, value=0, species=mobile_H),
]

my_model.settings = F.Settings(atol=1e10, rtol=1e-10, max_iterations=30, final_time=100)

my_model.settings.stepsize = F.Stepsize(initial_value=0.5)

right_flux = F.SurfaceFlux(field=mobile_H, surface=right_boundary)

my_model.exports = [right_flux]

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell}
:tags: [hide-input]

import plotly.graph_objects as go
import plotly.express as px

zeta = (6 * rho_W * np.exp((E_D - E_p) / (F.k_B * T)) + c_m * N_A) / (rho_W * 1e-3)
D_eff = D / (1 + 1 / zeta)

# plot computed solution
t = np.array(right_flux.t)
computed_solution = np.array(right_flux.data)

fig = go.Figure()

festim_plot = go.Scatter(
    x=t,
    y=computed_solution / 2 / 1e16,
    mode="lines",
    line=dict(width=3, color=px.colors.qualitative.Plotly[1]),
    name="FESTIM",
)

# plot exact solution
m = np.arange(1, 10001)

# Calculate the exponential part for all m values at once
tau = sample_depth**2 / (np.pi**2 * D_eff)
exp_part = np.exp(-(m**2) * t[:, None] / tau)

# Calculate the 'add' part for all m values and sum them up for each t
add = 2 * (-1) ** m * exp_part
exact_solution = 1 + add.sum(
    axis=1
)  # Sum along the m dimension and add 1 for the initial ones

exact_solution = N_A * exact_solution * c_m * D / (2 * sample_depth)

exact_plot = go.Scatter(
    x=t,
    y=exact_solution / 1e16,
    mode="markers",
    marker=dict(size=9, opacity=0.5, color=px.colors.qualitative.Plotly[0]),
    name="exact",
)

fig.add_traces([exact_plot, festim_plot])
fig.update_xaxes(title_text="Time, s", range=[0, 100])
fig.update_yaxes(
    title_text="Downstream flux, 10<sup>16</sup> H m<sup>-2</sup>s<sup>-1</sup>"
)
fig.update_layout(template="simple_white", height=600)

# The writing-reading block below is needed to avoid the issue with compatibility
# of Plotly plots and dollarmath syntax extension in Jupyter Book
# For mode details, see https://github.com/jupyter-book/jupyter-book/issues/1528
fig.write_html("./tmap_1da.html")
from IPython.display import HTML, display

display(HTML("./tmap_1da.html"))
```
