# MDM3-Mindear-Project

## Sound Map Models

Five simulated environments are currently available:

- **Single stationary source** — one fixed sound source.
- **Multiple stationary sources** — several fixed sound sources.
- **Single moving source** — one source moving across the space.
- **Distributed background noise** — multiple stationary sources with time-varying sound levels.
- **Concert venue** — a simulated indoor concert environment incorporating directional speakers, wall reflections and acoustic treatments.

Each model generates a sound map over a 2D grid.

### Concert Venue Simulation

The concert venue model is inspired by the O2 Academy Bristol and includes:

- Two directional speakers and a defined stage area.
- Inverse-square sound attenuation with distance.
- First- and second-order specular wall reflections.
- Frequency-dependent wall absorption using concrete and acoustic panel properties.
- Comparison of treated and untreated wall configurations.
- An audience-accessible area restricting microphone placement.

The model calculates unweighted broadband sound pressure levels (SPL) using six octave-band frequencies from 125 Hz to 4,000 Hz.

All sound levels are simulated rather than measured, and the venue geometry and acoustic parameters include modelling assumptions.

## Data Preparation

`extract_sound.py` provides functions to prepare simulated sound maps for interpolation.

`prepare_interpolation_data()` generates phone positions and extracts their sound measurements for different configurations.

Supported phone layouts:

- Random
- Grid
- Clustered

Supported microphone counts:

- 10
- 30
- 50
- 100
- 250

The number and arrangement of phones can be varied to investigate how measurement density and spatial distribution affect interpolation accuracy.

For the concert venue, microphone positions are restricted to the audience-accessible area.

## Example Workflow

`test_input_for_interpolate.py` demonstrates how to load a sound map and prepare data for interpolation.

The workflow is:

1. Select a sound map model.
2. Load the sound map, grid coordinates and any spatial restrictions.
3. Generate microphone positions using the selected count and layout.
4. Extract simulated sound measurements at microphone positions.
5. Obtain the true sound levels across the grid for comparison.
6. Pass the prepared data to an interpolation method.
7. Compare interpolated predictions against the simulated ground truth.

### Visualisation

The scripts support visualising:

- Simulated sound-level heat maps.
- Microphone positions overlaid on sound maps.
- Different microphone counts and spatial layouts.
- Concert venue geometry, including speakers, stage and wall treatments.
- Treated and untreated concert venue sound fields.

## Future Work

- Implement and compare four interpolation models, including Kriging and Gaussian process regression.
- Evaluate interpolation accuracy across different microphone counts and layouts.
- Investigate prediction uncertainty and its relationship to microphone placement.
- Compare interpolation performance across different simulated sound environments.
- Extend the simulations to stadiums and racetracks.
- Assess the potential for reconstructing realistic sound maps from distributed smartphone measurements.
