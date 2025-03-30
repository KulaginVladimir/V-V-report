---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: vv-festim-report-env-festim-2
  language: python
  name: python3
---

# Deuterium retention in tungsten

```{tags} 1D, TDS, trapping, transient
```

This validation case is a thermo-desorption spectrum measurement perfomed by Ogorodnikova et al. {cite}`ogorodnikova_deuterium_2003`.

Deuterium ions at 200 eV were implanted in a 0.5 mm thick sample of high purity tungsten foil (PCW).

The ion beam with an incident flux of $2.5 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for 400 s which corresponds to a fluence of $1.0 \times 10^{22} \ \mathrm{D \ m^{-2}}$

The diffusivity of tungsten in the FESTIM model is as measured by Frauenfelder {cite}`frauenfelder_permeation_1968`.

To reproduce this experiment, three traps are needed: 2 intrinsic traps and 1 extrinsic trap.
The extrinsic trap represents the defects created during the ion implantation.

The time evolution of extrinsic traps density $n_i$ expressed in $\text{m}^{-3}$ is defined as:
\begin{equation}
    \frac{dn_i}{dt} = \varphi_0\:\left[\left(1-\frac{n_i}{n_{a_{max}}}\right)\:\eta_a \:f_a(x)+\left(1-\frac{n_i}{n_{b_{max}}}\right)\:\eta_b \:f_b(x)\right]
\end{equation}

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input]

import festim as F
import ufl


class CustomValue(F.Value):
    def __init__(self, input_value, species_dependent_value=None):
        self.species_dependent_value = species_dependent_value or {}
        super().__init__(input_value)

    def convert_input_value(
        self, function_space=None, t=None, temperature=None, up_to_ufl_expr=False
    ):
        mesh = function_space.mesh
        x = ufl.SpatialCoordinate(mesh)
        arguments = self.input_value.__code__.co_varnames
        kwargs = {}
        if "t" in arguments:
            kwargs["t"] = t
        if "x" in arguments:
            kwargs["x"] = x
        if "T" in arguments:
            kwargs["T"] = temperature

        for name, species in self.species_dependent_value.items():
            kwargs[name] = species.concentration

        self.fenics_object = self.input_value(**kwargs)


class CustomSource(F.ParticleSource):

    def __init__(self, value, volume, species, species_dependent_value=None):
        self.species_dependent_value = species_dependent_value or {}
        super().__init__(value, volume, species)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = CustomValue(value, self.species_dependent_value)
```

```{code-cell} ipython3
:tags: [hide-output]

import festim as F
import numpy as np
import matplotlib.pyplot as plt

import ufl

model = F.HydrogenTransportProblem()

sample_depth = 5e-4

vertices = np.concatenate(
    [
        np.linspace(0, 30e-9, num=200),
        np.linspace(30e-9, 3e-6, num=300),
        np.linspace(3e-6, 20e-6, num=200),
        np.linspace(20e-6, sample_depth, num=100),
    ]
)

model.mesh = F.Mesh1D(vertices)

tungsten = F.Material(D_0=4.1e-07, E_D=0.39)

volume = F.VolumeSubdomain1D(id=1, borders=[0, sample_depth], material=tungsten)
left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
right_boundary = F.SurfaceSubdomain1D(id=2, x=sample_depth)

model.subdomains = [volume, left_boundary, right_boundary]


w_atom_density = 6.3e28  # atom/m3

H = F.Species("H")
trapped_H1 = F.Species("trapped_H1", mobile=False)
trapped_H2 = F.Species("trapped_H2", mobile=False)
trapped_H3 = F.Species("trapped_H3", mobile=False)
empty_trap1 = F.ImplicitSpecies(n=1.3e-3 * w_atom_density, others=[trapped_H1])
empty_trap2 = F.ImplicitSpecies(n=4e-4 * w_atom_density, others=[trapped_H2])
empty_trap3 = F.Species("empty_trap3", mobile=False)
model.species = [H, trapped_H1, trapped_H2, trapped_H3, empty_trap3]

trapping_reaction_1 = F.Reaction(
    reactant=[H, empty_trap1],
    product=[trapped_H1],
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=0.87,
    volume=volume,
)
trapping_reaction_2 = F.Reaction(
    reactant=[H, empty_trap2],
    product=[trapped_H2],
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.0,
    volume=volume,
)

trapping_reaction_3 = F.Reaction(
    reactant=[H, empty_trap3],
    product=[trapped_H3],
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.5,
    volume=volume,
)

model.reactions = [
    trapping_reaction_1,
    trapping_reaction_2,
    trapping_reaction_3,
]


imp_fluence = 1e22
incident_flux = 2.5e19  # beam strength from paper

imp_time = imp_fluence / incident_flux  # s

ion_flux = lambda t: ufl.conditional(t <= imp_time, incident_flux, 0)


def gaussian_distribution(x, center, width):
    return (
        1
        / (width * (2 * ufl.pi) ** 0.5)
        * ufl.exp(-0.5 * ((x[0] - center) / width) ** 2)
    )


particle_source = lambda x, t: ion_flux(t) * gaussian_distribution(x, 4.5e-9, 2.5e-9)

n_amax = 1e-1 * w_atom_density
n_bmax = 1e-2 * w_atom_density
eta_a = 6e-4
eta_b = 2e-4
f_b = lambda x: ufl.conditional(x[0] < 1e-6, 1e6, 0)

trap_generation_value = lambda x, t, n_empty: ion_flux(t) * (
    (1 - n_empty / n_amax) * eta_a * gaussian_distribution(x, 4.5e-9, 2.5e-9)
    + (1 - n_empty / n_bmax) * eta_b * f_b(x)
)
trap_source = CustomSource(
    value=trap_generation_value,
    volume=volume,
    species=empty_trap3,
    species_dependent_value={"n_empty": empty_trap3},
)
model.sources = [
    F.ParticleSource(value=particle_source, volume=volume, species=H),
    trap_source,
]

# boundary conditions
model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
    F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
]
implantation_temp = 293  # K
temperature_ramp = 8  # K/s

start_tds = imp_time + 50  # s


def temp_fun(t):
    if t <= start_tds:
        return implantation_temp
    else:
        return implantation_temp + temperature_ramp * (t - start_tds)


model.temperature = temp_fun

min_temp, max_temp = implantation_temp, 700


model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    final_time=start_tds
    + (max_temp - implantation_temp) / temperature_ramp,  # time to reach max temp
)

model.settings.stepsize = F.Stepsize(
    initial_value=0.5,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
    max_stepsize=lambda t: 0.5 if t >= start_tds else None,
    milestones=[start_tds],
)


left_flux = F.SurfaceFlux(surface=left_boundary, field=H)
right_flux = F.SurfaceFlux(surface=right_boundary, field=H)
total_mobile_H = F.TotalVolume(field=H, volume=volume)
trapped_H1 = F.TotalVolume(field=trapped_H1, volume=volume)
trapped_H2 = F.TotalVolume(field=trapped_H2, volume=volume)
trapped_H3 = F.TotalVolume(field=trapped_H3, volume=volume)

model.exports = [
    total_mobile_H,
    trapped_H1,
    trapped_H2,
    trapped_H3,
    left_flux,
    right_flux,
]

model.initialise()
model.run()
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

t = left_flux.t
flux_total = np.array(left_flux.data) + np.array(right_flux.data)

contribution_trap_1 = -np.diff(trapped_H1.data) / np.diff(trapped_H1.t)
contribution_trap_2 = -np.diff(trapped_H2.data) / np.diff(trapped_H2.t)
contribution_trap_3 = -np.diff(trapped_H3.data) / np.diff(trapped_H3.t)

t = np.array(t)
temp = implantation_temp + 8 * (t - start_tds)

# plotting simulation data
plt.plot(temp, flux_total, linewidth=3, label="FESTIM")

# plotting trap contributions
plt.plot(temp[1:], contribution_trap_1, linestyle="--", color="grey")
plt.fill_between(temp[1:], 0, contribution_trap_1, facecolor="grey", alpha=0.1)
plt.plot(temp[1:], contribution_trap_2, linestyle="--", color="grey")
plt.fill_between(temp[1:], 0, contribution_trap_2, facecolor="grey", alpha=0.1)
plt.plot(temp[1:], contribution_trap_3, linestyle="--", color="grey")
plt.fill_between(temp[1:], 0, contribution_trap_3, facecolor="grey", alpha=0.1)


# plotting original data
experimental_tds = np.genfromtxt("ogorodnikova-original.csv", delimiter=",")
experimental_temp = experimental_tds[:, 0]
experimental_flux = experimental_tds[:, 1]
plt.scatter(experimental_temp, experimental_flux, color="green", label="Experiment", s=16)

plt.legend()
plt.xlim(min_temp, max_temp)
plt.ylim(bottom=-1.25e18, top=0.6 * 1e19)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

plt.show()
```

```{note}
The experimental data was taken from Figure 5 of the original experiment paper {cite}`ogorodnikova_deuterium_2003` using [WebPlotDigitizer](https://automeris.io/)
```
