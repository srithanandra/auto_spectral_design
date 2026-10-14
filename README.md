# Autonomous Photonic Design: $\alpha\text{-MoO}_3$ Polaritonic Spectral Classifier

An autonomous, closed-loop inverse design pipeline for engineering phonon-polariton ($\text{PhP}$) scatterer geometries in biaxial $\alpha\text{-MoO}_3$ thin films[cite: 5, 13]. Inspired by [`flexcompute/autophotonicdesign`](https://github.com/flexcompute/autophotonicdesign), this project couples **Tidy3D** GPU-accelerated FDTD simulations with physical Design Rule Checking (DRC) to optimize class-specific nanophotonic sections that distinguish **Pentacene** from **Polystyrene** via near-field Photo-induced Force Microscopy (PiFM) readout[cite: 5, 12, 13].

---

## 1. Physical Concept & Optical Architecture

The optical classifier identifies unknown chemical analytes by probing the near-field interference of hyperbolic phonon-polaritons launched by sub-wavelength scatterers[cite: 5, 10]:

* **Polaritonic Substrate**: A $100\ \text{nm}$ ($\text{h\_moo3} = 0.1\ \mu\text{m}$) biaxial $\alpha\text{-MoO}_3$ crystal flake resting on a Silicon ($\varepsilon = 12$) substrate with the top surface at $z = 0.0\ \mu\text{m}$[cite: 13].
* **Anisotropic Lorentz Dispersion**: In-plane and out-of-plane permittivities ($\varepsilon_{\infty,x} = 5.8$, $\varepsilon_{\infty,y} = 6.1$, $\varepsilon_{\infty,z} = 4.5$) are modeled using multi-pole Lorentz oscillators derived from the LO-TO phonon product form (3 active phonon oscillators along $x$, 1 along $y$, and 1 along $z$)[cite: 13].Within the $860\text{--}920\ \text{cm}^{-1}$ spectral window, the crystal exhibits in-plane Type-II hyperbolic dispersion where $\text{Re}(\varepsilon_x) < 0 < \text{Re}(\varepsilon_y)$[cite: 13].
* **Scatterer Launchers**: Four circular apertures ($r = 0.1\ \mu\text{m}$, diameter $= 200\ \text{nm}$, length $= 0.105\ \mu\text{m}$) are etched through the $\alpha\text{-MoO}_3$ film at polar coordinates $(r_1, r_2, r_3, r_4, \theta_1, \theta_2, \theta_3, \theta_4)$[cite: 5, 13]. Each rim breaks translational invariance to supply the high in-plane momentum required to launch interfering PhP wavefronts[cite: 13].
* **Excitation & PiFM Near-Field Readout**: A normally incident, $E_x$-polarized plane wave at $z = 6.0\ \mu\text{m}$ (Gaussian pulse centered at $880\ \text{cm}^{-1}$, $\Delta f = f_0 / 10$) illuminates the structure along the $-z$ direction[cite: 13]. An AFM cantilever tip probes the resulting near-field electric field intensity ($\vert{}E_z\vert{}^2$) at a clearance of $z = 30\ \text{nm}$ ($0.03\ \mu\text{m}$) above the center of the flake across $N = 49$ discrete wavenumber points[cite: 5, 7, 13].

---

## 2. Classification Metric & Figure of Merit (FoM)

Classification is realized by reading out the near-field spectral response from class-specific sections and comparing the integrated intensity overlap with each analyte's vibrational fingerprint[cite: 6, 7, 12]:

1. **Total Integrated Intensity**: Across the $N = 49$ sampled wavenumbers $\lambda_i \in [860, 920]\ \text{cm}^{-1}$, the total intensity response for analyte $m \in \{\text{pentacene}, \text{polystyrene}\}$ with normalized absorption spectrum $K_m(\lambda_i)$ is[cite: 7, 12, 13]:
   $$I_{\text{total}}^{(m)} = \sum_{i=1}^{49} I_{\text{sim}}(\lambda_i) \cdot K_m(\lambda_i)$$

2. **Binary Selectivity Score ($\text{FoM}_c$)**: For a section dedicated to target chemical $c$, the optimization objective maximizes the relative intensity contrast between the target analyte and the competing analyte while preserving spectral shape alignment:
   $$\text{FoM}_c = 0.6 \cdot \left(\frac{I_{\text{total}}^{(c)}}{I_{\text{total}}^{(\text{pentacene})} + I_{\text{total}}^{(\text{polystyrene})} + \epsilon}\right) + 0.4 \cdot \left(\frac{\sum_{i=1}^{49} I_{\text{sim}}(\lambda_i) K_c(\lambda_i)}{\Vert{}I_{\text{sim}}\Vert{}_2 \Vert{}K_c\Vert{}_2}\right)$$

---

## 3. Repository Structure & File Contracts

```text
polariton_autodesign/
├── README.md          # Project documentation and execution guide
├── program.md         # System specifications, FDTD constants, and DRC constraints
├── targets.py         # 49-point wavenumber grid and Pentacene/Polystyrene IR profiles
├── design.py          # Lorentz MoO3 material builder and 4-scatterer slab constructor
├── drc.py             # Focused Ion Beam (FIB) and PiFM tip clearance validator
├── simulate.py        # Tidy3D FDTD runner, mesh override assertions, and field extractor
├── orchestrate.py     # Closed-loop evaluator, binary FoM scorer, and journal logger
├── journal.md         # Markdown log tracking iteration scores, intensities, and parameters
└── data/              # Directory storing Tidy3D .hdf5 simulation outputs and target CSVs
```

* **`design.py`**: The primary parameter interface. Accepts an 8-element array `[r1, r2, r3, r4, theta1, theta2, theta3, theta4]`, verifies the hyperbolic band condition ($\text{Re}(\varepsilon_x) < 0 < \text{Re}(\varepsilon_y)$), and performs boolean subtraction of the four cylinders from the $\alpha\text{-MoO}_3$ slab[cite: 13].
* **`drc.py`**: Prevents wasted cloud FDTD compute by enforcing three physical fabrication rules prior to simulation:
  * **PiFM Tip Exclusion Zone**: $r_n \ge 0.20\ \mu\text{m}$ so holes do not overlap the central AFM probing region.
  * **Section Footprint Bound**: $r_n \le 1.25\ \mu\text{m}$ to keep scatterers within the local section.
  * **FIB Milling Wall Clearance**: Minimum center-to-center separation of $0.25\ \mu\text{m}$ ($50\ \text{nm}$ minimum crystal wall thickness between $r = 0.1\ \mu\text{m}$ holes) to prevent lattice damage and hole merging[cite: 13].
* **`simulate.py`**: Constructs a $6.0 \times 6.0 \times 8.5\ \mu\text{m}$ FDTD domain with `Absorber` boundaries on all sides, `symmetry=(0, 0, 0)`, an enforced $30\ \text{nm}$ in-plane grid (`xy_30nm`), a $20\ \text{nm}$ vertical film override (`film_dz_20nm`), `run_time = 2e-11 s`, and `shutoff = 1e-7`[cite: 13].

---

## 4. Installation & Setup

### Prerequisites
* Python 3.13+[cite: 13]
* A [Flexcompute Tidy3D](https://tidy3d.simulation.cloud/) account and API key[cite: 13]

### Install Dependencies
```bash
pip install tidy3d numpy scipy matplotlib
```

### Configure Tidy3D Authentication
Set your Tidy3D API key in your environment or configure it directly via Python[cite: 13]:
```bash
tidy3d configure --apikey YOUR_TIDY3D_API_KEY
```

---

## 5. Usage

### Run the Baseline Evaluation
To verify your environment, run the baseline 4-scatterer square layout at $(\pm 0.5, \pm 0.5)\ \mu\text{m}$[cite: 13]:
```bash
python orchestrate.py
```
This script will:
1. Validate the 8 polar coordinates against `drc.py`.
2. Build the $\alpha\text{-MoO}_3$ Lorentz medium and assert the $30\ \text{nm} \times 20\ \text{nm}$ mesh overrides in `simulate.py`[cite: 13].
3. Submit the FDTD task to the Tidy3D cloud and save the result to `data/moo3_pentacene_iter_1.hdf5`[cite: 13].
4. Compute $I_{\text{total}}^{(\text{pentacene})}$, $I_{\text{total}}^{(\text{polystyrene})}$, and the binary $\text{FoM}$[cite: 7, 12].
5. Append the results to `journal.md`.

### Run in Autonomous Agent Mode
To run an autonomous optimization loop using an LLM coding agent (e.g., Claude Code, Cursor, or Aider):
1. Open your agent in the repository root.
2. Provide the following prompt:
   > *"Read `program.md` and `journal.md`. Your goal is to optimize the 8 scatterer parameters `[r1, r2, r3, r4, theta1, theta2, theta3, theta4]` in `orchestrate.py` first for `target_chemical='pentacene'` (Section 1) and then for `target_chemical='polystyrene'` (Section 2). Propose physically motivated coordinate updates that exploit in-plane hyperbolic polariton interference, verify they pass `drc.py`, execute `orchestrate.py`, and iterate until the binary selectivity FoM exceeds 0.75."*

### Custom Chemical Spectra
By default, `targets.py` generates calibrated 49-point Lorentzian absorption profiles for Pentacene and Polystyrene. To use experimental FTIR/NIST data, place two two-column CSV files (`wavenumber_cm_1, absorbance`) at:
* `data/pentacene.csv`
* `data/polystyrene.csv`

`BinaryChemicalTargets.load_or_generate_spectra(Path("data"))` will automatically interpolate and normalize them onto the 49-point simulation grid[cite: 7, 12].
