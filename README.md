

# GatorST: A Versatile Contrastive Meta-Learning Framework for Spatial Transcriptomic Data Analysis


## Requirements
- python : 3.9.12
- scanpy : 1.10.3
- sklearn : 1.1.1
- scipy : 1.9.0
- torch : 1.11.2
- torch-geometric : 2.1.0
- numpy : 1.24.4
- pandas : 2.2.3


## Project Structure

```bash
.
├── main.py            # Main training and evaluation loop
├── model.py           # Model architecture and loss functions
├── data_loader.py     # Data loading and graph construction utilities
├── util.py            # Utility functions (seed setup, metrics, dropout)
├── data/              # Folder for .h5ad input files
├── saved_models/      # Folder to save trained models
├── saved_graph/       # Folder for cached graphs and subgraphs
└── result.json        # Evaluation results output
```

## Usage

To train and evaluate on spatial transcriptomics datasets:

```bash
python main.py
```

Ensure your `.h5ad` data files are located in the `data/` directory.

## Outputs
- Trained models saved in `saved_models/`
- Intermediate subgraph representations saved in `saved_graph/`
- Final clustering results and metrics saved in `result.json`

## Datasets
- LIBD Human Dorsolateral Prefrontal Cortex (DL: http://research.libd.org/spatialLIBD/
- Human Breast Cancer: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0
- Mouse Brain Tissue: https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0

## Usage  
First, specify the hyper-parameters (e.g., number of epochs, batch size) in the file train.py. Then run the following command:
```bash
python train.py
```


## Citation
