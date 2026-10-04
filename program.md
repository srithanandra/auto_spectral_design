# Autonomous Design Specification: Binary Polaritonic Classifier (Pentacene vs. Polystyrene)

## 1. Physical Architecture & FDTD Parameters
* **Substrate & Film**: 100 nm ($\text{h\_moo3} = 0.1\ \mu\text{m}$) biaxial $\alpha\text{-MoO}_3$ flake on a Silicon ($\varepsilon = 12$) substrate with top surface at $z_{\text{top}} = 0.0\ \mu\text{m}$.
* **Dispersion Model**: Multi-oscillator Lorentz model converted from the LO-TO product form (3 oscillators along $x$, 1 along $y$, 1 along $z$), operating in the Type-II in-plane hyperbolic band ($\text{Re}(\varepsilon_x) < 0 < \text{Re}(\varepsilon_y)$).
* **Scatterer Geometry**: 4 circular scatterers ($r = 0.1\ \mu\text{m}$, length $= 0.105\ \mu\text{m}$) subtracted directly from the $\alpha\text{-MoO}_3$ slab geometry[cite: 13].
* **Excitation Source**: Normal-incidence $E_x$-polarized `PlaneWave` at $z = 6.0\ \mu\text{m}$ propagating in the $-z$ direction with a Gaussian pulse centered at $880\ \text{cm}^{-1}$ (`fwidth = freq0 / 10`)[cite: 13].
* **Domain & Meshing**: $6.0 \times 6.0\ \mu\text{m}$ lateral footprint, $z \in [-1.0, 7.5]\ \mu\text{m}$ ($L_z = 8.5\ \mu\text{m}$), `Absorber` boundaries on all sides, no mirror symmetry (`symmetry=(0,0,0)`), $30\ \text{nm}$ enforced in-plane mesh (`dx_xy = 0.03`), $20\ \text{nm}$ vertical film mesh (`dz = 0.02`), `run_time = 2e-11 s`, and `shutoff = 1e-7`[cite: 13].
* **Near-Field Readout**: `FieldMonitor` (`field_xy`) at $z = 0.03\ \mu\text{m}$ ($30\ \text{nm}$ above the film top surface), where $\vert{}E_z\vert{}^2$ (or $\vert{}E_x\vert{}^2$) is integrated over the central PiFM cantilever tip zone[cite: 5, 13].

## 2. Binary Target Analytes
1. **Class 1**: Pentacene
2. **Class 2**: Polystyrene[cite: 5]

## 3. Binary Classification Metric
* **Total Intensity Readout**: Across $N = 49$ sampled wavenumbers in the active hyperbolic band, the total intensity for chemical $m \in \{\text{pentacene}, \text{polystyrene}\}$ is:
  $$I_{\text{total}}^{(m)} = \sum_{i=1}^{49} I_{\text{sim}}(\lambda_i) \cdot K_m(\lambda_i)$$
* **Binary Selectivity Figure of Merit ($\text{FoM}_c$)**: For target class $c$ against competing class $\bar{c}$:
  $$\text{FoM}_c = 0.6 \cdot \left(\frac{I_{\text{total}}^{(c)}}{I_{\text{total}}^{(\text{pentacene})} + I_{\text{total}}^{(\text{polystyrene})} + \epsilon}\right) + 0.4 \cdot \left(\frac{\sum_{i=1}^{49} I_{\text{sim}}(\lambda_i) K_c(\lambda_i)}{\Vert{}I_{\text{sim}}\Vert{}_2 \Vert{}K_c\Vert{}_2}\right)$$