from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tidy3d as td
from targets import BinaryChemicalTargets


def extract_spectrum_from_hdf5(
    hdf5_path: Path,
    monitor_name: str = "field_xy",
    probe_radius: float = 0.20
) -> npt.NDArray[np.float64]:
    """Extracts the 1D near-field |Ez|^2 intensity spectrum from a Tidy3D SimulationData file."""
    try:
        sim_data: td.SimulationData = td.SimulationData.from_file(str(hdf5_path))
        field_mon = sim_data[monitor_name]
        ez_cropped = field_mon.Ez.sel(
            x=slice(-probe_radius, probe_radius),
            y=slice(-probe_radius, probe_radius)
        )
        ez_vals: npt.NDArray[np.complex128] = ez_cropped.values
        spectrum: npt.NDArray[np.float64] = np.mean(np.abs(ez_vals) ** 2, axis=(0, 1, 2))
        return spectrum / (np.max(spectrum) + 1e-12)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract spectrum from {hdf5_path}: {exc}") from exc


def plot_slide_spectral_classification(
    wavenumbers: npt.NDArray[np.float64],
    section_1_spectrum: npt.NDArray[np.float64],
    section_2_spectrum: npt.NDArray[np.float64],
    unknown_chemical: str = "pentacene",
    save_path: Path = Path("data/spectral_classification_plot.png")
) -> None:
    """Generates the two-panel near-field spectral readout and total intensity classification figure."""
    try:
        targets_mgr: BinaryChemicalTargets = BinaryChemicalTargets(
            lower_wn=float(np.min(wavenumbers)),
            upper_wn=float(np.max(wavenumbers)),
            num_points=len(wavenumbers)
        )
        profiles: dict[str, npt.NDArray[np.float64]] = targets_mgr.load_or_generate_spectra()
        if unknown_chemical not in profiles:
            raise ValueError(f"Unknown chemical '{unknown_chemical}' not found in target profiles.")
        analyte_profile: npt.NDArray[np.float64] = profiles[unknown_chemical]
        readout_sec_1: npt.NDArray[np.float64] = section_1_spectrum * analyte_profile
        readout_sec_2: npt.NDArray[np.float64] = section_2_spectrum * analyte_profile
        i_total_1: float = float(np.sum(readout_sec_1))
        i_total_2: float = float(np.sum(readout_sec_2))
        fig, (ax_spec, ax_bar) = plt.subplots(
            1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True
        )
        ax_spec.plot(
            wavenumbers, readout_sec_1, color="#2ca02c", linewidth=2.2, label="① Pentacene Section"
        )
        ax_spec.vlines(
            wavenumbers, 0.0, readout_sec_1, color="#2ca02c", linestyle="--", alpha=0.35
        )
        ax_spec.plot(
            wavenumbers, readout_sec_2, color="#1f77b4", linewidth=2.2, label="② Polystyrene Section"
        )
        ax_spec.vlines(
            wavenumbers, 0.0, readout_sec_2, color="#1f77b4", linestyle=":", alpha=0.35
        )
        ax_spec.plot(
            wavenumbers, analyte_profile, color="black", linestyle="-.", alpha=0.5, label=f"Analyte ({unknown_chemical})"
        )
        ax_spec.set_xlabel(r"$\omega, \mathrm{cm}^{-1}$", fontsize=12)
        ax_spec.set_ylabel("Intensity, a.u.", fontsize=12)
        ax_spec.set_title("Near-Field Spectral Response (N = 49 points)", fontsize=13)
        ax_spec.legend(loc="upper right")
        ax_spec.grid(True, alpha=0.25)
        bars = ax_bar.bar(
            ["① Pentacene", "② Polystyrene"],
            [i_total_1, i_total_2],
            color=["#2ca02c", "#1f77b4"],
            edgecolor="black",
            width=0.55
        )
        for bar_item in bars:
            height: float = float(bar_item.get_height())
            ax_bar.annotate(
                f"{height:.2f}",
                xy=(bar_item.get_x() + bar_item.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )
        ax_bar.set_ylabel(r"$\sum_{i=1}^{49} I_i(\lambda_i) = I_{\mathrm{total}}$ (a.u.)", fontsize=12)
        ax_bar.set_title(f"Classification (Input: {unknown_chemical.capitalize()})", fontsize=13)
        ax_bar.set_ylim(0.0, max(i_total_1, i_total_2) * 1.2 + 1e-6)
        ax_bar.grid(axis="y", alpha=0.25)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.show()
    except Exception as exc:
        raise RuntimeError(f"Failed to generate spectral classification plot: {exc}") from exc