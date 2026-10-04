import numpy as np
import numpy.typing as npt
import tidy3d as td


def convert_inverse_cent_to_hz(inverse_cent: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """Converts wavenumbers in cm^-1 to frequencies in Hz for Tidy3D."""
    try:
        return inverse_cent * td.C_0 / 1e4
    except Exception as exc:
        raise ValueError(f"Frequency conversion failed: {exc}") from exc


def lo_to_lorentz_coeffs(eps_inf: float, oscillators: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Converts LO-TO phonon product parameters into Tidy3D Lorentz pole-residue coefficients."""
    try:
        w_lo: npt.NDArray[np.float64] = np.array([osc[1] for osc in oscillators], dtype=np.float64)
        w_to: npt.NDArray[np.float64] = np.array([osc[2] for osc in oscillators], dtype=np.float64)
        coeffs: list[tuple[float, float, float]] = []
        for j, (gamma_cm, _, w_to_cm) in enumerate(oscillators):
            strength: float = float(eps_inf * (w_lo[j] ** 2 - w_to[j] ** 2))
            for k in range(len(oscillators)):
                if k != j:
                    strength *= float((w_lo[k] ** 2 - w_to[j] ** 2) / (w_to[k] ** 2 - w_to[j] ** 2))
            coeffs.append((
                float(strength / (w_to[j] ** 2)),
                float(convert_inverse_cent_to_hz(w_to_cm)),
                float(convert_inverse_cent_to_hz(gamma_cm)) / 2.0
            ))
        return coeffs
    except Exception as exc:
        raise RuntimeError(f"Lorentz coefficient calculation failed: {exc}") from exc


def get_moo3_medium() -> td.AnisotropicMedium:
    """Builds and validates the biaxial alpha-MoO3 AnisotropicMedium in the hyperbolic band."""
    try:
        eps_x_inf: float = 5.8
        eps_y_inf: float = 6.1
        eps_z_inf: float = 4.5
        osc_x: list[tuple[float, float, float]] = [
            (49.1, 534.3, 506.7),
            (6.0, 963.0, 821.4),
            (0.35, 999.2, 998.7),
        ]
        osc_y: list[tuple[float, float, float]] = [(9.5, 850.1, 544.6)]
        osc_z: list[tuple[float, float, float]] = [(1.5, 1006.9, 956.7)]
        moo3_x: td.Lorentz = td.Lorentz(eps_inf=eps_x_inf, coeffs=lo_to_lorentz_coeffs(eps_x_inf, osc_x))
        moo3_y: td.Lorentz = td.Lorentz(eps_inf=eps_y_inf, coeffs=lo_to_lorentz_coeffs(eps_y_inf, osc_y))
        moo3_z: td.Lorentz = td.Lorentz(eps_inf=eps_z_inf, coeffs=lo_to_lorentz_coeffs(eps_z_inf, osc_z))
        study_wn: npt.NDArray[np.float64] = np.array([860.0, 880.0, 900.0, 920.0], dtype=np.float64)
        study_f: npt.NDArray[np.float64] = np.asarray(convert_inverse_cent_to_hz(study_wn), dtype=np.float64)
        if not np.all(np.real(moo3_x.eps_model(study_f)) < 0.0):
            raise ValueError("Sanity check failed: Re(eps_x) must be negative in the 860-920 cm^-1 band.")
        if not np.all(np.real(moo3_y.eps_model(study_f)) > 0.0):
            raise ValueError("Sanity check failed: Re(eps_y) must be positive in the 860-920 cm^-1 band.")
        return td.AnisotropicMedium(xx=moo3_x, yy=moo3_y, zz=moo3_z)
    except Exception as exc:
        raise RuntimeError(f"Failed to create MoO3 medium: {exc}") from exc


def build_section_geometry(
    params: npt.NDArray[np.float64],
    scatter_radius: float = 0.1,
    h_moo3: float = 0.1
) -> tuple[list[td.Structure], list[tuple[float, float]]]:
    """Constructs the 100 nm MoO3 slab with 4 punched scatterers and the Si substrate."""
    try:
        if params.ndim != 1 or len(params) != 8:
            raise ValueError("Expected 1D parameter array of length 8: [r1, r2, r3, r4, theta1, theta2, theta3, theta4].")
        r_vals: npt.NDArray[np.float64] = params[:4]
        theta_vals: npt.NDArray[np.float64] = params[4:]
        inf_eff: float = 1e2
        z_top: float = 0.0
        film_box: td.Box = td.Box.from_bounds(
            rmin=(-inf_eff, -inf_eff, -h_moo3),
            rmax=(inf_eff, inf_eff, z_top)
        )
        scatter_centers: list[tuple[float, float]] = [
            (float(r_val * np.cos(t_val)), float(r_val * np.sin(t_val)))
            for r_val, t_val in zip(r_vals, theta_vals)
        ]
        scatterers: list[td.Cylinder] = [
            td.Cylinder(
                center=(x_pos, y_pos, z_top - h_moo3 / 2.0),
                radius=scatter_radius,
                length=h_moo3 * 1.05,
                axis=2
            )
            for x_pos, y_pos in scatter_centers
        ]
        film_geometry: td.Geometry = film_box
        for scatterer in scatterers:
            film_geometry = film_geometry - scatterer
        moo3_slab: td.Structure = td.Structure(
            geometry=film_geometry,
            medium=get_moo3_medium(),
            name="MoO3_film_4scatter"
        )
        si_slab: td.Structure = td.Structure(
            geometry=td.Box.from_bounds(
                rmin=(-inf_eff, -inf_eff, -inf_eff),
                rmax=(inf_eff, inf_eff, -h_moo3)
            ),
            medium=td.Medium(permittivity=12.0, name="Si"),
            name="Si_substrate"
        )
        return [moo3_slab, si_slab], scatter_centers
    except Exception as exc:
        raise RuntimeError(f"Geometry construction error: {exc}") from exc