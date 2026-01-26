# Efficient Cross-Architecture Knowledge Transfer for Large-Scale Online User Response Prediction

An efficient cross-architecture knowledge transfer framework for large-scale online user response prediction, supporting multiple model architectures and datasets.

## 📋 Table of Contents

- [Project Introduction](#project-introduction)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Experimental Results](#experimental-results)
- [References](#references)

## 📖 Project Introduction

CrossAdapt is a cross-architecture knowledge transfer framework specifically designed for large-scale online user response prediction tasks. The framework effectively transfers knowledge from large teacher models to student models through a two-stage training process, significantly reducing model transfer overhead while maintaining model performance.

### Core Features

- **Two-Stage Training Pipeline**:
  - **Stage 1**: Student model initialization and knowledge distillation
    - Stage 1A: Sparse embedding layer knowledge distillation
    - Stage 1B: Dense model layer knowledge distillation
    - Stage 1C: Joint training of embedding and dense model layers
  - **Stage 2**: Online learning and continuous distillation

- **Flexible Model Architecture Support**: Supports multiple CTR model architectures as teacher or student models
- **Diverse Sample Selection Strategies**: Supports various sample selection methods including random sampling, biased sampling, temporal sampling, etc.

## 🔧 Requirements

- Python >= 3.7
- PyTorch >= 2.0 (with CUDA support)
- Other dependencies are listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

1. **Train Teacher Model**:
```bash
python3 run.py --mode teacher
```

2. **Run Complete CrossAdapt Pipeline**:
```bash
python3 run.py --mode tskd --replay true --sample_selection biased_sample --sample_ratio 0.1
```

Or use the provided script:
```bash
bash run.sh
```

### Using Configuration File

The project supports configuration through the `config.yaml` file. All parameters can be set in the configuration file and can also be overridden by command-line arguments.

## ⚙️ Configuration

### Main Configuration Items

#### Basic Configuration (`basic`)
- `mode`: Running mode (`teacher`/`student`/`tskd`)
- `dataset`: Dataset selection (`avazu`/`criteo`/`criteo_ctr_full`)
- `num_runs`: Number of experimental runs (for result averaging)
- `stage2_update_mode`: Stage 2 training mode (`both`/`student`)

#### Model Architecture (`model_architecture`)
- `teacher_arch`: Teacher model architecture
- `student_arch`: Student model architecture
- `teacher_embedding_dim`: Teacher model embedding dimension
- `student_embedding_dim`: Student model embedding dimension
- `hidden_dims_teacher`: Teacher model hidden layer dimensions list
- `hidden_dims_student`: Student model hidden layer dimensions list

#### Stage 1 Parameters (`stage1`)
- `s1a_method`: Stage 1A method (default: `orthomap`)
- `alpha_1b`: Stage 1B distillation loss weight
- `alpha_1c`: Stage 1C distillation loss weight
- `s1_dense_epochs`: Stage 1B training epochs
- `s1_joint_epochs`: Stage 1C training epochs

#### Stage 2 Parameters (`stage2`)
- `replay`: Whether to enable replay mechanism
- `alpha_2`: Stage 2 distillation loss weight
- `teacher_update_freq`: Teacher model update frequency
- `replay_ratio`: Replay batch ratio

#### Sample Selection (`sample_selection`)
- `sample_selection`: Sample selection method
- `sample_ratio`: Representative sample ratio
- `target_pos_rate`: Target positive sample rate for biased sampling

For detailed configuration instructions, please refer to the comments in the `config.yaml` file.

## 📖 Usage

### Command-Line Arguments

All configuration parameters can be passed via command-line arguments, which will override the corresponding parameters in the configuration file.

Example:
```bash
python3 run.py \
    --mode tskd \
    --dataset criteo \
    --teacher_arch mlp \
    --student_arch fibi \
    --sample_selection biased_sample \
    --sample_ratio 0.1 \
    --replay true \
    --alpha_2 0.7
```

### Running Modes

1. **Train Teacher Model Only** (`--mode teacher`):
   - Train and save the teacher model
   - Can be used for pre-training or standalone evaluation of teacher model performance

2. **Train Student Model Only** (`--mode student`):
   - Train student model only (requires a pre-trained teacher model)

3. **Complete CrossAdapt Pipeline** (`--mode tskd`):
   - Execute the complete two-stage knowledge transfer pipeline
   - Includes teacher model training and student model distillation

## 📁 Project Structure

```
CrossAdapt/
├── README.md                 # Project documentation
├── run.py                    # Main running script
├── run.sh                    # Quick run script
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── result.txt                # Experimental results record
├── TSKD/                     # Core framework code (CrossAdapt implementation)
│   ├── __init__.py
│   ├── framework.py          # Framework main logic
│   ├── model.py              # Model definitions
│   ├── trainer.py            # Trainer
│   ├── dataset.py            # Dataset processing
│   └── dataset_stream.py     # Streaming dataset
├── criteo_logs/              # Training logs directory
├── criteo_saved_models/      # Saved models directory
├── criteo_loss_curves/       # Loss curve plots
└── criteo_processed_data/    # Processed data directory
```

## 📊 Experimental Results

Experimental results are saved in the `result.txt` file, containing the following metrics:
- AUC (Area Under Curve)
- LogLoss
- Training time

Example result format:
```
Average AUC, LogLoss, Time(s): 73.17±0.16,0.4042±0.0009,53.4±0.3
```

## 🔍 Output Files

- **Log Files** (`{dataset}_logs/`): Detailed logs of the training process
- **Model Checkpoints** (`{dataset}_saved_models/`): Saved model weights
- **Loss Curves** (`{dataset}_loss_curves/`): Training loss visualization plots


## 📝 Notes

1. **Data Preparation**: Ensure datasets are correctly placed in the project directory
2. **GPU Support**: GPU is recommended for training, as CPU training will be slower
3. **Memory Management**: For large datasets, pay attention to adjusting the `full_in_memory` parameter
4. **Configuration Override**: Command-line arguments override corresponding parameters in the configuration file

