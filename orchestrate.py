from pathlib import Path
import numpy as np
import numpy.typing as npt
from targets import BinaryChemicalTargets
from design import build_section_geometry
from drc import verify_drc
from simulate import create_fdtd_simulation, run_and_extract_spectrum


def compute_binary_metrics(
    sim_spectrum: npt.NDArray[np.float64],
    target_chemical: str,
    chemical_profiles: dict[str, npt.NDArray[np.float64]]
) -> tuple[float, dict[str, float]]:
    """Calculates I_total = sum_{i=1}^{49} I_i(lambda_i) for Pentacene and Polystyrene and scores selectivity."""
    try:
        if target_chemical not in ("pentacene", "polystyrene"):
            raise ValueError("target_chemical must be 'pentacene' or 'polystyrene'.")
        norm_sim: npt.NDArray[np.float64] = sim_spectrum / (np.max(sim_spectrum) + 1e-12)
        total_intensities: dict[str, float] = {
            chem_name: float(np.sum(norm_sim * chem_spec))
            for chem_name, chem_spec in chemical_profiles.items()
        }
        target_intensity: float = total_intensities[target_chemical]
        sum_both: float = total_intensities["pentacene"] + total_intensities["polystyrene"] + 1e-12
        selectivity_ratio: float = target_intensity / sum_both
        target_vec: npt.NDArray[np.float64] = chemical_profiles[target_chemical]
        cosine_overlap: float = float(
            np.dot(norm_sim, target_vec) / (np.linalg.norm(norm_sim) * np.linalg.norm(target_vec) + 1e-12)
        )
        fom: float = 0.6 * selectivity_ratio + 0.4 * cosine_overlap
        return fom, total_intensities
    except Exception as exc:
        raise RuntimeError(f"Failed to compute binary classification metrics: {exc}") from exc


def append_to_journal(
    iteration: int,
    target_chemical: str,
    fom: float,
    intensities: dict[str, float],
    params: npt.NDArray[np.float64],
    journal_path: Path = Path("journal.md")
) -> None:
    """Logs the iteration score, Pentacene vs. Polystyrene total intensities, and geometry parameters."""
    try:
        if not journal_path.exists():
            header: str = (
                "| Iter | Target | FoM | I_Pentacene | I_Polystyrene | Parameters |\n"
                "|---|---|---|---|---|---|\n"
            )
            journal_path.write_text(header, encoding="utf-8")
        row: str = (
            f"| {iteration} | {target_chemical} | {fom:.4f} | "
            f"{intensities.get('pentacene', 0.0):.3f} | "
            f"{intensities.get('polystyrene', 0.0):.3f} | "
            f"{np.round(params, 3).tolist()} |\n"
        )
        with journal_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(row)
    except Exception as exc:
        raise RuntimeError(f"Failed to write to journal: {exc}") from exc


def evaluate_section_design(
    params: npt.NDArray[np.float64],
    target_chemical: str = "pentacene",
    iteration: int = 1
) -> float:
    """Executes DRC, Tidy3D FDTD simulation, and Pentacene vs. Polystyrene scoring for a candidate design."""
    try:
        targets_mgr: BinaryChemicalTargets = BinaryChemicalTargets(num_points=49)
        profiles: dict[str, npt.NDArray[np.float64]] = targets_mgr.load_or_generate_spectra()
        _, centers = build_section_geometry(params)
        passed, msg = verify_drc(centers)
        if not passed:
            print(f"[DRC REJECTED] {msg}")
            return -1.0
        sim = create_fdtd_simulation(params, targets_mgr.wavenumbers)
        sim_spectrum: npt.NDArray[np.float64] = run_and_extract_spectrum(
            sim,
            task_name=f"moo3_{target_chemical}_iter_{iteration}",
            run_path=f"data/moo3_{target_chemical}_iter_{iteration}.hdf5"
        )
        fom, intensities = compute_binary_metrics(sim_spectrum, target_chemical, profiles)
        append_to_journal(iteration, target_chemical, fom, intensities, params)
        return fom
    except Exception as exc:
        print(f"Evaluation failed at iteration {iteration}: {exc}")
        return -1.0


if __name__ == "__main__":
    baseline_r: float = float(np.hypot(0.5, 0.5))
    baseline_params: npt.NDArray[np.float64] = np.array(
        [baseline_r, baseline_r, baseline_r, baseline_r, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
        dtype=np.float64
    )
    score: float = evaluate_section_design(baseline_params, target_chemical="pentacene", iteration=1)
    print(f"Baseline (x,y = +-0.5 um) Pentacene Selectivity FoM: {score:.4f}")