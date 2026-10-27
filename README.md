# MDM3-Mindear-Project

## Sound Map Models
Four simulated environments are currently available:

- **Single stationary source** — one fixed sound source.
- **Multiple stationary sources** — several fixed sound sources.
- **Single moving source** — one source moving across the space.
- **Distributed background noise** — multiple stationary sources with time-varying sound levels.

Each model generates a sound map over a 2D grid.

## Data Preparation
`extract_sound.py` provides functions to prepare simulated sound maps for interpolation.

`prepare_interpolation_data()` generates phone positions and extracts their sound measurements for different configurations.

Supported phone layouts:
- Random
- Grid
- Clustered

The number of phones can also be varied to investigate how measurement density affects interpolation accuracy.

## Example Workflow
`test_input_for_interpolate.py` demonstrates how to load a sound map and prepare data for an interpolation method.

The workflow is:
1. Select a sound map model.
2. Load the sound map and grid coordinates.
3. Generate phone positions and extract their sound measurements.
4. Obtain the true sound levels across the grid for comparison.
5. Pass the prepared data to an interpolation method.

## Future Work
- Input prepared data and loaded sound maps into 4 different interpolation models.
- Compare interpolation accuracy across different phone counts and layouts.
- Combine models and adjust arrangement of sound sources to simulate more complex environments, such as concerts, stadiums, or racetracks.
