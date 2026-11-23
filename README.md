# MDM3-Mindear-Project

Sound field reconstruction for the MindEar SoundTracker app: simulated venue sound maps are sampled by virtual "phones", and four interpolation models are compared on how well they rebuild the full map from those readings.

## Repository structure

```
MDM3-Mindear-Project/
├── sound_maps/
│   ├── basic/          early, simple sound fields (point sources, moving source, background noise)
│   └── venues/         final venue simulations (concert, stadium, F1) and phone sampling
├── models/
│   ├── idw/            Inverse Distance Weighting
│   ├── kriging/        Ordinary Kriging (no buffer / buffer / intensity-aware buffer)
│   ├── gaussian_process/   Gaussian Process regression (RBF kernel)
│   └── diffusion_lms/  Diffusion LMS (distributed estimation across phones)
└── calibration/        phone calibration experiment (wind tunnel) data and processing
```

Every script and notebook finds the repo root by itself, so they can be run from the repo root or opened from their own folder.

## Sound maps

### `sound_maps/venues/` (current pipeline)

| File | Purpose |
|---|---|
| `final_concert_map.py` | Concert venue: directional speakers, wall reflections, acoustic treatment. |
| `final_stadium_map.py` | Football stadium: time-varying crowd noise, chanting, goal celebrations, reflections. |
| `final_race_track_map.py` | F1 race track, driven by `final_synchronised_f1.csv`. |
| `final_single_wall.py` | Shared wall geometry and acoustic material properties. |
| `final_extract_sound_venues.py` | Places phones and extracts their readings (perfect phones). |
| `bias_and_variance_extract_sound_venues.py` | Same as above but adds measured phone error (bias −2.59 dB, SD 1.89 dB). |
| `final_visualise_microphone_configurations.py` | Plots the phone layouts over a sound map. |
| `final_test_input_for_interpolate_venue.py` | Minimal example of preparing the inputs for an interpolation model. |

Each venue's `load_sound_map()` returns `sound_maps`, `x`, `y` and `allowed_mask` (where phones can be placed).

`prepare_interpolation_data(...)` takes these plus phone counts (**10, 30, 50, 100, 250**) and layouts (**random, grid, clustered**) and returns, for each `(count, layout)`:

- `phone_positions` — (N, 2) phone coordinates
- `phone_levels` — (T, N) readings in dB, one row per time step
- `query_positions` — (M, 2) audience locations to predict
- `true_levels` — (T, M) simulated truth, for evaluation only
- `evaluation_mask` — query points that are not phone locations (score models on these)

### `sound_maps/basic/` (early development)

`point_source.py`, `multiple_point_sources.py`, `single_moving_point_source.py`, `distributed_background_noise.py`, with `extract_sound.py` for sampling them.

## Models

All four models take the same inputs, `phone_positions (N,2)`, `phone_levels (T,N)`, `query_positions (M,2)`, and return predicted levels `(T, M)` in dB.

| Folder | Model code | Testing / results |
|---|---|---|
| `models/idw/` | `idw_model.py` — `idw(...)` and `choose_p_k(...)` (leave-one-out tuning of power `p` and neighbours `k`) | `run_idw.py` (basic maps), `run_idw_venue.py` (concert + stadium), `test_idw.py` (pytest) |
| `models/kriging/` | `Ordinary_Kriging_Model_no_buffer.py`, `_buffer.py`, `_buffer_diffusion.py` (intensity-aware safety cushion) | `Ordinary_Kriging_Model_Test_Runs.ipynb`, `all_predicted_sound_maps.png` |
| `models/gaussian_process/` | `gp_all_model.py` (RBF GP), `gp_kernel_and_predict.py` | `gp_all_sampling.py` → `gp_all_model.py` → `gp_all_test.py` (run in that order), `gp_all_model.ipynb`, `gp_sampling.py`, `gp_tast.py`, `gp_working_final_*.ipynb` |
| `models/diffusion_lms/` | `diffusion_lms_model.py` — `DiffusionAcousticLMS` | `diffusion_fianl_testing.ipynb`, `Diffusion_lms_model_testing.ipynb`, `diffusion_final_testing_results.csv` |

`idw(...)` and the kriging class in `Ordinary_Kriging_Model_buffer_diffusion.py` take a `domain` option: `"energy"` (default) averages readings as linear energy (10^(L/10)); `"db"` averages the dB values directly.

### Running

From the repo root, for example:

```bash
python3 models/idw/run_idw_venue.py
python3 -m pytest models/idw/test_idw.py
python3 models/gaussian_process/gp_all_sampling.py
python3 models/gaussian_process/gp_all_model.py
python3 models/gaussian_process/gp_all_test.py
```

Notebooks can be opened from anywhere inside the repo. Scripts save their figures and outputs next to themselves.

## Calibration

`calibration/` holds the phone calibration experiment: `SoundTracker_Calibration_Kit/` (raw recordings, processed data, analysis summaries, reference meter log, protocol notes) and `Data_Processing_Script.ipynb`, which processes it (run from inside `calibration/`). Older zipped snapshots of the kit are in `calibration/archives/`.

## Known issues

- `gp_working_final_f1.ipynb`, `gp_working_final_maps_concert.ipynb` and `gp_working_final_stadium.ipynb` import `gp_working_final_*.py` scripts that are not in the repo, so they only show saved outputs and cannot be re-run yet.
- Cell 19 of `Ordinary_Kriging_Model_Test_Runs.ipynb` imports `Ordinary_Kriging_Model`, which no longer exists; it needs pointing at one of the three kriging files.
