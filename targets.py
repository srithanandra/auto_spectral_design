from pathlib import Path
import numpy as np
import numpy.typing as npt


class BinaryChemicalTargets:
    """Manages the 49-point wavenumber grid and target IR spectra for Pentacene and Polystyrene."""

    def __init__(self, lower_wn: float = 860.0, upper_wn: float = 920.0, num_points: int = 49) -> None:
        try:
            self.wavenumbers: npt.NDArray[np.float64] = np.linspace(lower_wn, upper_wn, num_points, dtype=np.float64)
            self.chemical_names: tuple[str, str] = ("pentacene", "polystyrene")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize BinaryChemicalTargets: {exc}") from exc

    def load_or_generate_spectra(self, data_dir: Path | None = None) -> dict[str, npt.NDArray[np.float64]]:
        """Loads 49-point Pentacene and Polystyrene spectra from CSVs or generates calibrated profiles."""
        try:
            spectra: dict[str, npt.NDArray[np.float64]] = {}
            if data_dir is not None and data_dir.exists():
                for name in self.chemical_names:
                    csv_path: Path = data_dir / f"{name}.csv"
                    if csv_path.exists():
                        raw_data: npt.NDArray[np.float64] = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
                        interp_spec: npt.NDArray[np.float64] = np.interp(self.wavenumbers, raw_data[:, 0], raw_data[:, 1])
                        spectra[name] = interp_spec / (np.max(interp_spec) + 1e-12)
            if len(spectra) != 2:
                spectra = self._default_binary_profiles()
            return spectra
        except Exception as exc:
            raise RuntimeError(f"Error loading or generating binary chemical spectra: {exc}") from exc

    def _default_binary_profiles(self) -> dict[str, npt.NDArray[np.float64]]:
        """Generates synthetic 49-point IR vibrational profiles for Pentacene and Polystyrene."""
        try:
            wn: npt.NDArray[np.float64] = self.wavenumbers
            peaks: dict[str, list[tuple[float, float, float]]] = {
                "pentacene": [(904.0, 5.5, 1.0), (872.0, 4.5, 0.55)],
                "polystyrene": [(906.0, 6.5, 0.75), (888.0, 6.0, 1.0)]
            }
            profiles: dict[str, npt.NDArray[np.float64]] = {}
            for name, modes in peaks.items():
                signal: npt.NDArray[np.float64] = np.zeros_like(wn, dtype=np.float64)
                for center, gamma, amp in modes:
                    signal += amp * (gamma ** 2) / ((wn - center) ** 2 + gamma ** 2)
                profiles[name] = signal / (np.max(signal) + 1e-12)
            return profiles
        except Exception as exc:
            raise RuntimeError(f"Error constructing binary benchmark profiles: {exc}") from exc