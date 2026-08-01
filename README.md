# Mamba-BiSAMNet

Official implementation of **Mamba-BiSAMNet: An ECG Denoising Network for Multi-Type Noise Removal**.

Mamba-BiSAMNet is a one-dimensional U-Net-based ECG denoising model. It combines residual convolutional blocks with a Bidirectional Structure-Aware Mamba (BiSAM) module in the mid-level skip connection. The repository contains the model, data preparation scripts, training procedure, and evaluation metrics used for synthetic ECG denoising experiments.

The released mid-level BiSAM configuration contains approximately 0.670 million trainable parameters with the standard Mamba defaults.

The current default configuration in `main.py` trains and evaluates the model under motion artifact (MA) noise using two channel- and segment-swapped noise versions, NV1 and NV2.

## Repository Structure

```text
Mamba-BiSAMNet-code/
|-- .gitignore
|-- config/
|   `-- base.yaml
|-- data/
|-- dataset/
|-- Data_Preparation/
|   |-- Prepare_QTDatabase.py
|   |-- Prepare_NSTDB.py
|   |-- data_preparation.py
|   `-- mixed_data_making.py
|-- model/
|   `-- mamba_bisamnet.py
|-- trainandtest/
|   `-- trainandtest.py
|-- utils/
|   |-- helpers.py
|   `-- metrics.py
|-- main.py
|-- README.md
`-- requirements.txt
```

The directories have the following roles:

- `data/`: original PhysioNet databases and intermediate files produced during preprocessing.
- `dataset/`: processed noisy-clean datasets and record-boundary files.
- `Data_Preparation/`: QTDB and NSTDB preprocessing and noise synthesis.
- `model/`: Mamba-BiSAMNet architecture.
- `trainandtest/`: training, validation, checkpointing, and evaluation.
- `utils/`: random-seed utilities and evaluation metrics.

## Environment

A Python environment with PyTorch and Mamba-SSM is required. The default configuration uses `cuda:0`, so install a PyTorch build compatible with the CUDA version on the target system.

Install PyTorch by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

Mamba-SSM installation requirements may depend on the PyTorch, CUDA, and operating-system versions. See the [official Mamba repository](https://github.com/state-spaces/mamba) for installation details.

## Original Data

The original databases are not included in this repository. Download and extract them from PhysioNet.

### QT Database

Download QTDB from:

https://www.physionet.org/content/qtdb/1.0.0/

### MIT-BIH Noise Stress Test Database

Download NSTDB from:

https://www.physionet.org/content/nstdb/1.0.0/

After extraction, the directory names must match the paths below:

```text
data/
|-- qt-database-1.0.0/
`-- mit-bih-noise-stress-test-database-1.0.0/
```

## Data Preparation

Run all commands from the repository root.

QTDB recordings are resampled from 250 Hz to 360 Hz to match NSTDB. ECG beats are extracted using QTDB annotations and inserted into 512-sample windows with a fixed 16-sample leading offset. Resampled beats longer than 496 samples are excluded because they do not fit in the window after this offset. This implementation yields 13,316 retained test heartbeats from the fixed 14-record test set. The remaining records are retained for record-independent training and validation splitting.

The preparation scripts generate the following intermediate files:

```text
data/QTDatabase.pkl
data/NoiseBWL.pkl
```

They also generate the record-boundary files required by `main.py`:

```text
dataset/flag_train.npy
dataset/train_record_names.npy
```

`flag_train.npy` stores the starting heartbeat index of each development record. This allows `main.py` to split training and validation data at the record level instead of splitting individual heartbeats.

### Single-Noise Datasets

The single-noise preparation function accepts `BW`, `MA`, or `EM` and a noise version of `1` or `2`.

BW and MA scaling factors are sampled reproducibly in 0.01 increments from `[0.20, 2.00)`. EM uses the lower range `[0.10, 1.00)` because of its larger native amplitude. The preparation function resets its NumPy seed to 1234, so regenerating a dataset with the same inputs produces the same scale sequence.

For each NSTDB noise record, both channels are divided into two non-overlapping temporal halves. NV1 uses the first half of channel 1 for training and the second half of channel 2 for testing. NV2 uses the first half of channel 2 for training and the second half of channel 1 for testing. The two versions are therefore trained as separate datasets and are not independent random realizations of the same noise segment.

Generate the MA datasets used by the current default training configuration:

```bash
python -c "from Data_Preparation.data_preparation import Data_Preparation; Data_Preparation('MA', 1)"
python -c "from Data_Preparation.data_preparation import Data_Preparation; Data_Preparation('MA', 2)"
```

The resulting files are:

```text
dataset/dataset_ma_nv1.pkl
dataset/dataset_ma_nv2.pkl
```

Replace `MA` with `BW` or `EM` to construct the corresponding single-noise datasets.

### Mixed-Noise Datasets

Generate mixed-noise datasets with:

```bash
python -c "from Data_Preparation.mixed_data_making import Data_Preparation; Data_Preparation(1)"
python -c "from Data_Preparation.mixed_data_making import Data_Preparation; Data_Preparation(2)"
```

The resulting files are:

```text
dataset/dataset_mixed_nv1.pkl
dataset/dataset_mixed_nv2.pkl
```

All generated dataset variants must use the same QTDB preprocessing and record order. The scripts verify existing record-boundary and record-name files and stop if an inconsistency is detected.

## Configuration

The default settings are defined in `config/base.yaml`:

| Setting | Default |
|---|---:|
| Device | `cuda:0` |
| Training batch size | 96 |
| Test batch size | 400 |
| Learning rate | 0.001 |
| Maximum epochs | 100 |
| AdamW weight decay | 0.01 |
| ExponentialLR gamma | 0.95 |
| Early-stopping patience | 10 epochs |

The composite training loss uses MSE, cosine-similarity loss, and total-variation loss with weights 0.7, 0.2, and 0.1, respectively.

## Training and Evaluation

After generating the required dataset files, run:

```bash
python main.py --config config/base.yaml --noise-types ma
```

The default command runs MA with both NV1 and NV2. Other prepared conditions can be selected without editing the source code:

```bash
python main.py --noise-types bw
python main.py --noise-types em
python main.py --noise-types mixed
python main.py --noise-types bw ma em mixed
```

Use `--versions nv1`, `--versions nv2`, or `--versions nv1 nv2` to select dataset versions. The default is both versions. Use `--num-seeds` to change the number of randomly generated training seeds; the default is five.

Each invocation randomly selects five distinct seeds and reuses the same five seeds for every selected noise condition and dataset version. For every seed, development records are split into training and validation subsets at the record level with an approximately 7:3 ratio. The code searches for a split whose validation heartbeat proportion is between 0.25 and 0.35. Heartbeats from one QTDB record therefore remain entirely within either training or validation.

NV1 and NV2 are trained separately. Their held-out test results are evaluated both individually and in combination.

## Outputs

Training and evaluation artifacts are written under:

```text
results/mamba_bisamnet/
|-- seed_results.json
`-- <noise_type>/
    `-- seed_<seed>/
        |-- model_params/
        |-- <noise_type>_nv1/
        |-- <noise_type>_nv2/
        `-- evaluation_results_combined.xlsx
```

For each dataset version, the code saves:

- Best and final model checkpoints.
- Noisy, denoised, and clean test signals in a NumPy file.
- Evaluation results in an Excel file.

The reported metrics are sum of squared differences (SSD), maximum absolute deviation (MAD), percentage root-mean-square difference (PRD), cosine similarity, input SNR, output SNR, and SNR improvement.

The randomly selected seeds, per-seed combined results, and the mean and sample standard deviation across completed seeds are stored in `results/mamba_bisamnet/seed_results.json`.

## Data Citations

The QTDB preprocessing procedure in this repository follows the DeepFilter-based protocol. Please cite:

Romero, F. P., Pinol, D. C., & Vazquez-Seisdedos, C. R. (2021). DeepFilter: An ECG baseline wander removal filter using deep learning techniques. *Biomedical Signal Processing and Control*, 70, 102992.

If you use the QT Database, please cite:

Laguna, P., Mark, R. G., Goldberg, A., et al. (1997). A database for evaluation of algorithms for measurement of QT and other waveform intervals in the ECG. In *Computers in Cardiology 1997* (pp. 673-676). IEEE.

If you use the MIT-BIH Noise Stress Test Database, please cite:

Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. *Computers in Cardiology*, 11, 381-384.

Please also cite PhysioNet:

**Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z. Available from: https://rdcu.be/faatM**

## Intended Use

This repository is intended for research on synthetic ECG denoising. The model has not been established as a clinical diagnostic system and should not be used for clinical decision-making without appropriate external validation.

## License

This project is released under the MIT License. See `LICENSE` for details.
