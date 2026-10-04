import numpy as np
import numpy.typing as npt
import tidy3d as td
import tidy3d.web as web
from design import build_section_geometry, convert_inverse_cent_to_hz


def create_fdtd_simulation(
    params: npt.NDArray[np.float64],
    wavenumbers: npt.NDArray[np.float64],
    h_moo3: float = 0.1
) -> td.Simulation:
    """Configures and validates the Tidy3D FDTD simulation matching MoO3_Permittivity.ipynb."""
    try:
        structures, _ = build_section_geometry(params, h_moo3=h_moo3)
        freqs: npt.NDArray[np.float64] = np.asarray(convert_inverse_cent_to_hz(wavenumbers), dtype=np.float64)
        freq0: float = float(convert_inverse_cent_to_hz(880.0))
        z_top: float = 0.0
        source_z: float = 6.0
        monitor_z: float = 0.03
        lx: float = 6.0
        ly: float = 6.0
        z_min: float = z_top - 1.0
        z_max: float = source_z + 1.5
        lz: float = z_max - z_min
        center_z: float = 0.5 * (z_min + z_max)
        run_time: float = 2e-11
        dx_xy: float = 0.03
        plane_wave: td.PlaneWave = td.PlaneWave(
            center=(0.0, 0.0, source_z),
            size=(td.inf, td.inf, 0.0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
            direction="-",
            pol_angle=0.0,
            name="plane_wave_Ex"
        )
        field_xy: td.FieldMonitor = td.FieldMonitor(
            center=(0.0, 0.0, monitor_z),
            size=(td.inf, td.inf, 0.0),
            freqs=freqs.tolist(),
            name="field_xy"
        )
        field_xz: td.FieldMonitor = td.FieldMonitor(
            center=(0.0, 0.0, z_top),
            size=(td.inf, 0.0, td.inf),
            freqs=freqs.tolist(),
            name="field_xz"
        )
        flux_r: td.FluxMonitor = td.FluxMonitor(
            center=(0.0, 0.0, source_z + 0.5),
            size=(td.inf, td.inf, 0.0),
            freqs=freqs.tolist(),
            name="flux_R",
            normal_dir="+"
        )
        xy_mesh: td.MeshOverrideStructure = td.MeshOverrideStructure(
            geometry=td.Box(center=(0.0, 0.0, center_z), size=(td.inf, td.inf, td.inf)),
            dl=(dx_xy, dx_xy, None),
            enforce=True,
            name="xy_30nm"
        )
        film_mesh: td.MeshOverrideStructure = td.MeshOverrideStructure(
            geometry=td.Box(center=(0.0, 0.0, z_top - h_moo3 / 2.0), size=(td.inf, td.inf, h_moo3)),
            dl=(None, None, 0.02),
            name="film_dz_20nm"
        )
        sim: td.Simulation = td.Simulation(
            center=(0.0, 0.0, center_z),
            size=(lx, ly, lz),
            sources=[plane_wave],
            structures=structures,
            monitors=[field_xy, field_xz, flux_r],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Absorber()),
            grid_spec=td.GridSpec.auto(
                wavelength=float(td.C_0 / freqs[0]),
                min_steps_per_wvl=10,
                override_structures=[xy_mesh, film_mesh]
            ),
            run_time=run_time,
            symmetry=(0, 0, 0),
            shutoff=1e-7
        )
        dx_arr: npt.NDArray[np.float64] = np.array(sim.grid.sizes.x, dtype=np.float64)
        if np.median(dx_arr) >= 0.04:
            raise ValueError("dx is still coarse - mesh override did not apply!")
        if [s.name for s in sim.structures] != ["MoO3_film_4scatter", "Si_substrate"]:
            raise ValueError("Unexpected structures in simulation.")
        return sim
    except Exception as exc:
        raise RuntimeError(f"Failed to assemble Tidy3D simulation: {exc}") from exc


def run_and_extract_spectrum(
    sim: td.Simulation,
    task_name: str = "moo3_pw_4scatter_perm_dx30nm",
    run_path: str = "data/moo3_pw_4scatter_perm_dx30nm.hdf5",
    probe_radius: float = 0.20
) -> npt.NDArray[np.float64]:
    """Submits the simulation to Tidy3D cloud and extracts the near-field |Ez|^2 spectrum at the tip."""
    try:
        job: web.Job = web.Job(simulation=sim, task_name=task_name, verbose=True)
        sim_data: td.SimulationData = job.run(path=run_path)
        field_data = sim_data["field_xy"]
        ez_cropped = field_data.Ez.sel(
            x=slice(-probe_radius, probe_radius),
            y=slice(-probe_radius, probe_radius)
        )
        ez_vals: npt.NDArray[np.complex128] = ez_cropped.values
        intensity_spectrum: npt.NDArray[np.float64] = np.mean(np.abs(ez_vals) ** 2, axis=(0, 1, 2))
        return intensity_spectrum.astype(np.float64)
    except Exception as exc:
        raise RuntimeError(f"Tidy3D cloud execution or extraction failed: {exc}") from exc