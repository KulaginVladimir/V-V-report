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

# Diffusion from a Depleting Source

```{tags} 1D, MES, transient
```

+++

This verification case {cite}`ambrosek_verification_2008` consists of an enclosure containing an initial quantity of hydrogen gas (at an initial pressure $P_0$).
The hydrogen can dissociate on the inner surface of the enclosure and then permeate through the walls.
As hydrogen particles escape the enclosure, the pressure decreases and as a result, so does the surface concentration on the inner walls.
This 1D problem is, therefore, coupled.
At each time step, the flux of particles escaping the enclosure is computed, and the internal pressure is updated.

An analytical solution to this problem can be obtained.
The analytical expression of the concentration in the wall is given by Ambrosek _et al_ {cite}`ambrosek_verification_2008`:


\begin{align}
    c(x=0, t) &= 0\\
    c(x=l, t) &= K_H \ P(t) \\
    c(x,t) &= 2 L P_{0} K_H \sum_{n=1}^{\infty} \frac{e^{- D t \alpha_n^2} \sin{\left(x \alpha_n \right)}}{\left(L + l \left(L^{2} + \alpha_n^2\right)\right) \sin{\left(l \alpha_n \right)}}
\end{align}
where $L = \dfrac{K_H \ T \ A \ k_B}{V}$, $K_H$ is the solubility of the material, $T$ is the temperature, $A$ is the enclosure surface area, $V$ is the volume of the enclosure, $l$ is the thickness of the wall, and $\alpha_n$ are the roots of:
\begin{equation}
    \alpha_n = \frac{L}{\tan(\alpha_n \ l)}
\end{equation}

The flux at the non-exposed surface can be expressed as:

\begin{align}
    J(x=0, t) &= D \nabla c |_{x=0} \nonumber
    \\ &= 2 L P_0 D K_H \sum_{n=1}^{\infty} \frac{e^{- D t \alpha_n^2} \alpha_n}{\left(L + l \left(L^{2} + \alpha_n^2\right)\right) \sin{\left(l \alpha_n \right)}}
\end{align}

The cumulative release $R$ is then:

\begin{align}
    \mathrm{R} &= \int_0^t J(x=0, t) dt\nonumber \\ &= 2 L P_0 S \sum_{n=1}^{\infty} \frac{1 - e^{- D t \alpha_n^{2}}}{\left(L + l \left(L^{2} + \alpha_n^2\right)\right) \alpha_n\sin{\left(l \alpha_n \right)}}
\end{align}

Multiplying by the area and normalising by the initial quantity in the enclosure, one can obtain the fractional release:

\begin{align}
    \mathrm{FR} &= \frac{\mathrm{R} \ A}{P_0 \ V  / (k_B T)} \nonumber\\
    &= \frac{2 L S A}{V / (k_B T)} \sum_{n=1}^{\infty} \frac{1 - e^{- D t \alpha_n^{2}}}{\left(L + l \left(L^{2} + \alpha_n^2\right)\right) \alpha_n\sin{\left(l \alpha_n \right)}}
\end{align}

+++

## FESTIM Code

```{code-cell} ipython3
import festim as F
import numpy as np

encl_vol = 5.20e-11  # m3  same
encl_surf = 2.16e-6  # m2  same
l = 3.3e-5  # m same
R = 8.314  # same
avogadro = 6.022e23  # mol-1  same
temperature = 2373  # K  same
initial_pressure = 1e6  # Pa  same
solubility = 7.244e22 / temperature  # H/m3/Pa  # same
diffusivity = 2.6237e-11  # m2/s  almost same


class PressureExport(F.SurfaceQuantity):
    def __init__(self):
        super().__init__(field=F.Species(), surface=left_surface)

    def compute(self):
        self.value = left_bc.value / solubility
        self.data.append(self.value)


class CustomHydrogenTransportProblem(F.HydrogenTransportProblem):
    def iterate(self):
        super().iterate()

        # Update pressure based on flux
        left_flux_val = -left_flux.value
        old_pressure = left_bc.value / solubility
        new_pressure = (
            old_pressure
            - (left_flux_val * encl_surf / encl_vol * R * self.temperature / avogadro)
            * self.dt.value
        )

        left_bc.value = new_pressure * solubility

        self.bc_forms[0] = self.create_dirichletbc_form(left_bc)


model = CustomHydrogenTransportProblem()

vertices = np.linspace(0, l, 50)

model.mesh = F.Mesh1D(vertices)

material = F.Material(D_0=diffusivity, E_D=0.0)

left_surface = F.SurfaceSubdomain1D(id=1, x=0)
right_surface = F.SurfaceSubdomain1D(id=2, x=l)
volume = F.VolumeSubdomain1D(id=3, borders=[0, l], material=material)
model.subdomains = [left_surface, volume, right_surface]

H = F.Species("H")
model.species = [H]

left_bc = F.DirichletBC(
    subdomain=left_surface, value=initial_pressure * solubility, species=H
)
model.boundary_conditions = [
    left_bc,
    F.DirichletBC(subdomain=right_surface, value=0, species=H),
]

model.temperature = temperature

left_flux = F.SurfaceFlux(field=H, surface=left_surface)
right_flux = F.SurfaceFlux(field=H, surface=right_surface)
pressure_export = PressureExport()

tot_vol = F.TotalVolume(field=H, volume=volume)
model.exports = [left_flux, right_flux, pressure_export]

model.settings = F.Settings(atol=1e8, rtol=1e-10, final_time=140)

model.settings.stepsize = F.Stepsize(0.1)

model.initialise()
model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/28766692/how-to-find-the-intersection-of-two-graphs/28766902#28766902

k = 1.38065e-23  # J/mol Boltzmann constant


def get_roots(L, l, alpha_max, step=0.0001):
    """Gets the roots of alpha = L / tan(alpha * l)

    Args:
        L (float): parameter L
        l (float): parameter l
        alpha_max (float): the maximum alpha to consider
        step (float, optional): the step discretising alphas.
            The smaller the step, the more accurate the roots.
            Defaults to 0.0001.

    Returns:
        np.array: array of roots
    """
    alphas = np.arange(0, alpha_max, step=step)[1:]

    f = alphas

    g = L / np.tan(alphas * l)

    # plt.plot(alphas, f, "-")
    # plt.plot(alphas, g, "-")

    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

    # remove one every other idx
    idx = idx[::2]
    # plt.plot(alphas[idx], f[idx], "ro")
    # plt.show()
    roots = alphas[idx]
    return roots


def get_roots_bis(L, alpha_max, step=0.0001):
    """Gets the roots of alpha = L / tan(alpha)

    Args:
        L (float): parameter L
        alpha_max (float): the maximum alpha to consider
        step (float, optional): the step discretising alphas.
            The smaller the step, the more accurate the roots.
            Defaults to 0.0001.

    Returns:
        np.array: array of roots
    """
    alphas = np.arange(0, alpha_max, step=step)[1:]

    f = alphas

    g = L / np.tan(alphas)

    plt.plot(alphas, f, "-")
    plt.plot(alphas, g, "-")

    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

    # remove one every other idx
    idx = idx[::2]
    plt.plot(alphas[idx], f[idx], "ro")
    plt.show()
    roots = alphas[idx]
    return roots


def analytical_expression_fractional_release_TMAP7(t, P_0, D, S, V, T, A, l):
    """
    FR = 1 - P(t) / P_0
    where P(t) is the pressure at time t and P_0 is the initial pressure

    Reference: https://doi.org/10.13182/FST05-A967 (Equation 4)
    Note: in the report, the expression of FR is given as P(T)/P_0, but it shown as 1 - P(t)/P_0 in the graph (Figure 1)

    Args:
        t (float, ndarray): time (s)
        P_0 (float): initial presure (Pa)
        D (float): diffusivity (m2/s)
        S (float): solubility (H/m3/Pa)
        V (float): enclosure volume (m3)
        T (float): temperature (K)
        A (float): enclosure surface area (m2)
        l (float): slab length (m)
    """
    L = S * T * A * k / V

    roots = get_roots(L=L, l=l, alpha_max=1e6, step=1)
    roots = roots[:, np.newaxis]
    summation = np.exp(-(roots**2) * D * t) / (l * (roots**2 + L**2) + L)
    summation = np.sum(summation, axis=0)

    pressure = 2 * P_0 * L * summation
    fractional_release = 1 - pressure / P_0
    return fractional_release


# ------------ post processing ----------------

t = right_flux.t
right_flux_data = right_flux.data
pressures = np.array(pressure_export.data)
fractional_release = 1 - pressures / initial_pressure

times = np.linspace(0, 140, 1000)

plt.figure()
plt.plot(t, fractional_release, label="FESTIM")

times = np.linspace(0, model.settings.final_time, 1000)
analytical = analytical_expression_fractional_release_TMAP7(
    t=times,
    P_0=initial_pressure,
    D=diffusivity,
    S=solubility,
    V=encl_vol,
    T=temperature,
    A=encl_surf,
    l=l,
)
plt.plot(times, analytical, label="analytical", color="tab:green", linestyle="--", lw=3)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Fractional release")
plt.gca().spines[["right", "top"]].set_visible(False)

plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

def analytical_expression_flux(t, P_0, D, S, V, T, A, l):
    """
    value of the flux at the external surface (not in contact with pressure)
    J = -D * dc/dx

    Args:
        t (float, ndarray): time (s)
        P_0 (float): initial presure (Pa)
        D (float): diffusivity (m2/s)
        S (float): solubility (H/m3/Pa)
        V (float): enclosure volume (m3)
        T (float): temperature (K)
        A (float): enclosure surface area (m2)
        l (float): slab length (m)
    """
    L = S * T * A * k / V

    roots = get_roots(L=L, l=l, alpha_max=1e7, step=1)
    roots = roots[:, np.newaxis]

    summation = (np.exp(-(roots**2) * D * t) * roots) / (
        (l * (roots**2 + L**2) + L) * np.sin(roots * l)
    )
    summation = np.sum(summation, axis=0)

    flux = 2 * S * P_0 * L * D * summation
    flux[0] = 0
    return flux


plt.plot(t, right_flux_data, label="FESTIM")
plt.plot(
    times,
    analytical_expression_flux(
        t=times,
        P_0=initial_pressure,
        D=diffusivity,
        S=solubility,
        V=encl_vol,
        T=temperature,
        A=encl_surf,
        l=l,
    ),
    color="tab:green",
    linestyle="--",
    label="analytical",
    lw=3,
)
plt.legend()
plt.ylim(bottom=0)
plt.ylabel("Flux at outer surface (H/m^2/s)")
plt.xlabel("Time (s)")

plt.gca().spines[["right", "top"]].set_visible(False)
plt.ylim(bottom=0)
plt.show()
```
