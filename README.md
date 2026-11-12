# RHIC BES ML Calibration

Machine learning workflows for calibrating the **integrated HydroKinetic Model (iHKM)** against observables from the **RHIC Beam Energy Scan (BES)** program.

The scientific objective is to constrain iHKM parameters using simulated and experimental hadronic observables. This repository implements two complementary approaches: neural networks that estimate parameters directly from observables, and neural network forward emulators coupled to Markov chain Monte Carlo (MCMC) inference. Energy-specific Jupyter notebooks and tabulated datasets cover √sₙₙ = 7.7, 11.5, 14.5, 19.6, 27, and 39 GeV.

The repository provides analysis workflows; it does not include the iHKM simulation engine or a packaged command-line application. No accuracy or convergence guarantee is implied by the included analyses.

## Repository structure

```text
.
├── data/                 # Simulation observables, parameter tables, and experimental inputs
├── src/
│   ├── ML/               # Six iHKMe_spectra_<energy>_v1.9.0.ipynb notebooks
│   └── MCMC/             # Six iHKMe_MCMC_<energy>_v1.7.0.ipynb notebooks
├── requirements.txt      # Python analysis dependencies
└── README.md
```

Each notebook is a standalone workflow for one collision energy. The MCMC notebooks train their own emulators and do not require running the ML notebooks first.

## Methodology

### Observables and parameters

Both approaches use six particle-spectrum blocks, ordered as π⁻, π⁺, K⁻, K⁺, p, and p̄, followed by elliptic flow, v₂. The data also contain HBT radius blocks (`Rout`, `Rside`, `Rlong`), but these are excluded from the networks' inputs or prediction targets in the current workflows.

The six inferred parameters, in output order, are:

| Column | Physical interpretation |
| --- | --- |
| `tau_0` | Initial proper time |
| `tau_rel` | Relaxation time |
| `tau_th` | Thermalization time |
| `eta_s` | Shear viscosity to entropy density ratio, η/s |
| `e_sw` | Switching energy density |
| `Rgauss` | Gaussian spatial scale parameter |

Parameter CSV files also contain `id` and `eos`; neither is among the six regression targets. Units, detailed parameter conventions, spectrum normalization, bin coordinates, and centrality selections are not specified in the supplied tables. Values are used in their stored numerical conventions; these details should be established before interpreting results physically.

### Direct parameter estimation: `src/ML/`

- Spectrum bins are transformed with the natural logarithm; v₂ remains in its native space. Each observable block and the six targets are standardized using training data only.
- Seven input branches each contain a 15-unit dense layer. Their concatenated representations pass through shared layers of 100, 50, and 20 units, with leaky ReLU activations, batch normalization, and dropout at rate 0.1.
- Twelve linear outputs encode six normalized parameter means and six signed scales. The active custom loss, `error_loss2`, combines squared mean error with a penalty matching squared scales to squared residuals; it is not a Gaussian negative log-likelihood.
- An ensemble of 50 networks is trained with Adam (learning rate 0.001), batch size 64, and 3,000 epochs per member. The current training loop has no early stopping.
- Five reproducible validation splits are defined using seeds 101, 211, 307, 401, and 503. Each holds out 100 rows while keeping identical six-parameter tuples together. One split is trained per run, selected by `ACTIVE_SPLIT_SEED` (default: 101).
- Ensemble predictions separate learned residual variance and variation between members. Monte Carlo propagation uses 2,000 experimental draws and 200 noisy closure draws per validation sample. Spectrum errors are approximated in log space as σ/x; v₂ errors are sampled in native space. Sampling treats bins independently.

The notebooks report validation and experimental estimates with uncertainty components. These diagnostic uncertainty estimates require assessment before scientific use.

### Forward emulation and calibration: `src/MCMC/`

The forward networks map the six standardized parameters to standardized log-spectra and native-space v₂. Each has hidden layers of 64, 128, and 128 units with swish activations and a linear observable output. Training uses mean squared error, Adam (learning rate 0.001), batch size 64, and up to 3,000 epochs, with early stopping after 100 epochs without validation improvement and restoration of the best weights.

Four models are trained for each of five grouped validation splits, giving 20 emulators per energy. Scalers are fitted separately to each split's training subset; predictions are mapped to common coordinates before aggregation.

Calibration uses `emcee` with a diagonal Gaussian likelihood combining experimental errors, validation-residual emulator errors, and ensemble spread. Experimental spectrum errors are propagated through the log transform using σ/x. The prior is uniform within the loaded parameter design's componentwise bounds, subject to:

```text
tau_0 + tau_rel <= tau_th < tau_0 + 3 * tau_rel
```

The default `STABLE` mode uses 32 walkers and 5,000 steps, discarding 1,000 burn-in steps and retaining every tenth step. Outputs include posterior medians and central 68% intervals, the maximum a posteriori sampled point, trace and corner plots, acceptance fractions, an autocorrelation check, and a comparison of the MAP prediction with experiment. These diagnostics must be reviewed to assess convergence and fit quality.

## Datasets

All files reside in [`data/`](data/). Energy tokens use hyphens in text filenames (`19-6`) and underscores in parameter CSV filenames (`19_6`).

| File pattern | Format and role |
| --- | --- |
| `ihkm_data_<energy>_v*.txt` | Headerless, tab-separated simulation observables; one simulation row per parameter point |
| `*_ML_new*.csv` | Comma-separated parameter tables with header `id,tau_0,tau_rel,tau_th,eta_s,e_sw,Rgauss,eos` |
| `exp_data_<energy>_v3.txt` | One headerless, tab-separated row of experimental observables |
| `err_data_<energy>_v3.txt` | One corresponding row of errors, treated as raw-space one-standard-deviation errors by the code |
| `cuts_<energy>_v3.txt` | One row of ten cumulative, exclusive column boundaries for the observable blocks |

The ten blocks follow the order of the six particle species, v₂, and the three HBT radii listed above. The last cut is the full row width; `cuts[5]` ends the spectra and `cuts[6]` ends the observables used by the networks. Full rows contain 151–160 bins depending on energy.

### Dataset versions selected by the notebooks

The table gives filename suffixes for `ihkm_data_` and the complete parameter filename. Row counts are the usable simulation rows expected by the current code, before holding out 100 validation rows.

| √sₙₙ (GeV) | ML simulation suffix | MCMC simulation suffix | Parameter CSV (ML / MCMC) | Usable rows (ML / MCMC) |
| --- | --- | --- | --- | --- |
| 7.7 | `7-7_v6.txt` | `7-7_v6.txt` | `7_7_ML_new_v4.csv` / same | 1,500 / 1,500 |
| 11.5 | `11-5_v3.txt` | `11-5_v3.txt` | `11_5_ML_new.csv` / same | 1,000 / 1,000 |
| 14.5 | `14-5_v4.txt` | `14-5_v4.txt` | `14_5_ML_new.csv` / same | 1,000 / 1,000 |
| 19.6 | `19-6_v4.txt` | `19-6_v5.txt` | `19_6_ML_new.csv` / `19_6_ML_new_v2.csv` | 1,000 / 1,290 |
| 27 | `27-0_v5.txt` | `27-0_v5.txt` | `27_0_ML_new.csv` / same | 1,500 / 1,500 |
| 39 | `39-0_v4.txt` | `39-0_v5.txt` | `39_0_ML_new.csv` / same | 1,000 / 1,299 |

Notebooks remove simulation rows containing missing values and select parameter rows using the retained original row indices. They rely on **row alignment**, rather than joining on `id`. Parameter tables can contain more rows than the selected simulation files. The 39 GeV MCMC loader limits both tables to its configured `DATA_ROWS = 1299` before filtering, using the first 1,299 of the 1,300 stored simulation rows. Do not reorder or independently filter paired files.

Other stored versions, including `ihkm_data_14-5_v3.txt`, are not selected by the current notebooks. The supplied files do not provide bibliographic provenance for the experimental measurements or a detailed simulation-generation record.

## Requirements and installation

[`requirements.txt`](requirements.txt) specifies the analysis dependencies: TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, emcee, corner, and tqdm. The latter supports the MCMC progress display. Dependencies are unpinned, so this file does not reproduce an exact historical environment.

All notebooks record Python **3.12.3** in their metadata. This is a recorded kernel version, not a declared minimum or evidence of tested compatibility. No supported Python version range is established in the repository; use an interpreter supported by the dependency versions you install.

For a local environment on Linux or macOS:

```bash
git clone https://github.com/NVlad1/rhic-bes-ml-calibration.git
cd rhic-bes-ml-calibration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

### Notebook tools

A notebook frontend and IPython kernel are not listed in `requirements.txt`. Use an existing notebook-capable environment, or install JupyterLab and a kernel in the activated environment:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --sys-prefix --name rhic-bes-ml-calibration --display-name "RHIC BES ML Calibration"
jupyter lab
```

JupyterLab is one option; another Jupyter-compatible frontend can use the same kernel.

### CPU and optional GPU execution

The notebooks use TensorFlow without requiring a GPU or selecting a CUDA device. CPU execution is the default setup described above. For optional NVIDIA GPU acceleration, follow the platform-specific [official TensorFlow pip installation guide](https://www.tensorflow.org/install/pip). GPU compatibility and performance have not been verified for this repository.

## Running an analysis

1. Launch your notebook frontend from the repository and select the environment created above.
2. Open an energy-specific notebook, for example [`src/ML/iHKMe_spectra_19-6_v1.9.0.ipynb`](src/ML/iHKMe_spectra_19-6_v1.9.0.ipynb) for direct estimation, or [`src/MCMC/iHKMe_MCMC_19-6_v1.7.0.ipynb`](src/MCMC/iHKMe_MCMC_19-6_v1.7.0.ipynb) for emulator-based calibration.
3. Review dataset selections and training settings. For ML, select `ACTIVE_SPLIT_SEED`; for MCMC, review `MODE` before the sampling cell.
4. Execute cells in order from a fresh kernel, then inspect the validation and inference diagnostics.

Data discovery searches the kernel's current working directory and its parents for a directory containing both `data/` and `src/`. Execution therefore works from the repository root or a notebook subdirectory. Starting from the root is convenient, but is not required.

Both workflows write split archives to `artifacts/<notebook-stem>/split_definitions/` relative to the **kernel working directory**. Archives preserve training/validation indices and their original source indices; ML also writes `split_audit.csv`. Predictions, plots, trained models, and MCMC chains otherwise remain in notebook output or memory unless explicitly saved by the user. No pretrained model files are supplied.

Default runs involve substantial training: 50 networks per ML run, or 20 networks followed by MCMC sampling. Reducing settings for a smoke run changes the analysis configuration. Fixed seeds and the ML notebooks' attempt to enable deterministic TensorFlow operations aid repeatability, but identical results across platforms and dependency versions are not established. Record the environment and settings when reporting results.

## Acknowledgments

This research project is supported by the **National Research Foundation of Ukraine (NRFU)** under grant No. 2025.07/0050.
