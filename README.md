# BaSFiN: Bayesian Skill–Feature Interaction Network for NBA Match Prediction

Bayesian Skill Factor Network for Team-Based Sports (2013–2024, 2009–2024).  
This repository contains the official implementation of **BaSFiN**, a Bayesian skill–factor model designed to estimate cooperative and competitive player contributions in team sports.

The project includes model implementations, preprocessing pipelines, training scripts, extended datasets, and hyperparameter search procedures used in the accompanying research.

---

## Installation

Install dependencies:

    pip install -r code/requirements.txt

---

## Project Structure

The project is organized as follows:

    BaSFiN_code/
    ├── code/         # Main program code and model implementations
    ├── data/         # Raw and processed feature datasets
    ├── NAC/          # Baseline model used for comparison in the paper
    ├── output/       # Output results
    └── plot/         # Visualization scripts

---

## 1. code/

When opening the project in VS Code, set **code/** as the working directory.  
It contains the following subfolders and modules.

### logs/

Stores experimental results and training logs.

### model/

Model storage and loading directory.  
Usually does not require modification.

### BaSFiN/

Core model and execution scripts for experiments using data from **2013–2024**.

Includes:

- `BaS.py` — main architecture  
- `co_fim.py` — cooperative feature interaction (pairwise score inspection version)  
- `co_fim2.py` — cooperative feature interaction (standard version)  
- `bc_fim.py` — competitive feature interaction (pairwise score inspection version)  
- `bc_fim2.py` — competitive feature interaction (standard version)  

Each module includes the corresponding training code and model definition scripts.

### BT/ and NAC/

Baseline models implemented for comparison in the research:

- **BT/** — Bradley–Terry model  
- **NAC/** — Neural Additive Composition model  

### BaSFiN_2009_2024/

Extended version of the model, enabling experiments on data from **2009–2024**.

### search/

Hyperparameter search scripts for identifying optimal parameters on the 2013–2024 dataset.

### processing/

Scripts for:

- Web crawling  
- Data table generation  
- Feature tensor construction  

Includes:

- `player_id_mapping_2009_2024.csv` — mapping table linking player IDs to yearly identifiers.

---

## 2. data/

This directory primarily includes:

- Statistical feature CSV files  
  - Used to generate feature tensors.  

- Team roster CSV files  
  - Used for match outcome prediction.

Data files themselves are not included in the public repository due to size and licensing considerations.

---

## Execution Workflow

### Step 1 — Set working directory

Open VS Code and set the working directory to:

    BaSFiN_code/code/

---

### Step 2 — Pretraining

Run pretraining scripts to independently optimize the cooperative and competitive modules:

    pretrain_fim.py

Pretraining stabilizes subsequent joint training of the BaSFiN model.

---

### Step 3 — Main Training

Run:

    train_basfin_noInter.py

Important argument:

    force_no_freeze = False

This controls whether certain parts of the model are kept frozen during training.

---

### Step 4 — Extended-Year Training (2009–2024)

To extend experiments to 2009–2024, switch to:

    BaSFiN_2009_2024/

Then run:

    train_basfin_noInter.py

The workflow is identical to the 2013–2024 version, but the dataset and tensor construction cover earlier seasons.

---

### Step 5 — Hyperparameter Search

Execute the scripts under:

    search/

This performs grid / iterative hyperparameter search on the 2013–2024 dataset.

---

## Notes

- The `model/` directory typically requires no modification unless model weights need to be replaced or updated.  
- The `logs/` directory will grow quickly with experiments; periodic cleanup is recommended.  
- Different modules (`co_fim`, `bc_fim`, etc.) correspond to different experimental designs; choose according to research needs.  
- Pretraining is strongly recommended before running the full BaSFiN training pipeline.  
- Data files must be placed correctly under `data/` before executing any scripts.
