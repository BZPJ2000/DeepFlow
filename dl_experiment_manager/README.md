# Deep Learning Experiment Manager

A dynamic web-based application for deep learning experiment management and visualization, integrated with PyTorch deep learning framework.

## Features

- **Dynamic Model Selection**: Browse and select from a wide range of deep learning architectures
- **Dataset Configuration**: Support for built-in datasets and custom data loading
- **Experiment Setup**: Configure training parameters, evaluation metrics, and experiment schedules
- **Performance Comparison**: Compare models across multiple datasets with statistical analysis
- **Academic Visualization**: Generate publication-ready charts, tables, and figures
- **No Code Repetition**: Dynamic loading mechanism eliminates repetitive script writing

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.13 or higher
- CUDA-capable GPU (optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/dl-experiment-manager.git
cd dl-experiment-manager
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the framework path:
   Edit `app.py` and update the `framework_path` variable to point to your deep learning framework:
   ```python
   framework_path = r"E:\Projects\Learning_space\2025_learn\torch-template-for-deep-learning-main"
   ```

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
dl_experiment_manager/
├── app.py                          # Main application entry point
├── requirements.txt                  # Dependencies
├── config/                         # Configuration files
├── core/                           # Core functionality
│   ├── dynamic_loader.py            # Dynamic class and function loading
│   └── config_manager.py           # Configuration management
├── pages/                          # Streamlit pages
│   ├── 1_Model_Selection.py         # Model selection interface
│   ├── 2_Dataset_Configuration.py  # Dataset configuration
│   ├── 3_Experiment_Setup.py       # Experiment setup
│   ├── 4_Performance_Comparison.py # Performance comparison
│   └── 5_Visualization_Dashboard.py # Visualization dashboard
├── utils/                          # Utility functions
├── static/                         # Static resources
├── templates/                      # HTML templates
└── docs/                           # Documentation
```

## Usage Guide

### 1. Model Selection

Navigate to the **Model Selection** page to:
- Browse available models by category (Classic Networks, Attention Networks, Lightweight Networks, etc.)
- Configure model parameters (number of classes, input channels, etc.)
- Preview model architecture and statistics
- Add models to your selection for comparison

**Tips**:
- Start with a few models for quick comparison
- Check model compatibility with your task
- Review model parameters before adding to selection

### 2. Dataset Configuration

Navigate to the **Dataset Configuration** page to:
- Select from built-in datasets (CIFAR10, ImageNet, COCO, etc.)
- Upload custom datasets from your local machine
- Configure data loading parameters (batch size, workers, etc.)
- Set up data augmentation options
- Configure train/val/test splits

**Tips**:
- Use data augmentation to improve generalization
- Adjust batch size based on GPU memory
- Ensure proper data splitting for unbiased evaluation

### 3. Experiment Setup

Navigate to the **Experiment Setup** page to:
- Configure training parameters (epochs, learning rate, optimizer, etc.)
- Select evaluation metrics (accuracy, precision, recall, F1, etc.)
- Set up learning rate scheduling
- Configure early stopping and checkpointing
- Create experiment combinations (all combinations, manual selection, or sequential)

**Tips**:
- Use learning rate scheduling for better convergence
- Enable early stopping to prevent overfitting
- Save checkpoints regularly to avoid data loss
- Set random seed for reproducibility

### 4. Performance Comparison

Navigate to the **Performance Comparison** page to:
- View summary statistics and rankings
- Create various visualizations (bar charts, line charts, radar charts, heatmaps)
- Perform statistical tests (t-test, ANOVA, Wilcoxon)
- Filter and export results

**Tips**:
- Use multiple visualization types for comprehensive analysis
- Statistical tests help validate significance
- Export results for publication

### 5. Visualization Dashboard

Navigate to the **Visualization Dashboard** page to:
- Generate interactive charts for presentations
- Create academic-quality tables for publications
- Generate publication-ready figures (PNG, PDF, SVG)
- Export in various formats with custom settings

**Tips**:
- Use high DPI (300+) for publications
- Export in PDF for vector graphics
- Follow journal formatting guidelines
- Use colorblind-friendly palettes

## Integration with Deep Learning Framework

The application integrates with the PyTorch deep learning framework located at:
```
E:\Projects\Learning_space\2025_learn\torch-template-for-deep-learning-main
```

### Supported Framework Components

- **Models**: All models in the `models/` directory
- **Datasets**: Datasets from the `dataloder.py` module
- **Loss Functions**: Losses from the `losses/` directory
- **Metrics**: Evaluation metrics from the `metrics/` directory
- **Optimizers**: Optimizers from the `optimizer/` directory

### Dynamic Loading

The application uses a dynamic loading mechanism to:
- Automatically discover available models, datasets, and functions
- Load classes and functions without code modification
- Instantiate objects with custom parameters
- Cache loaded components for performance

## Configuration Management

### Saving Configurations

Experiment configurations can be saved and loaded for reuse:
1. Configure your experiment in the relevant pages
2. Click "Save Configuration" button
3. Configuration is saved as JSON in the `configs/` directory

### Loading Configurations

1. Navigate to the page where you want to load a configuration
2. Click "Load Configuration" button
3. Select from the list of saved configurations

### Exporting Configurations

Configurations can be exported to:
- **JSON**: For data exchange and backup
- **Python Script**: For direct execution
- **LaTeX**: For documentation and publication

## Visualization Features

### Chart Types

- **Bar Charts**: Compare metrics across models
- **Line Charts**: Training progress over epochs
- **Radar Charts**: Multi-metric comparison
- **Heatmaps**: Performance matrices
- **Box Plots**: Metric distributions

### Academic Figures

- **Publication-Ready Bar Charts**: High DPI, publication styling
- **Multi-Panel Figures**: Multiple subplots in one figure
- **Model Architecture Diagrams**: Visual representation of model structure
- **Results Overview Figures**: Summary statistics and comparisons

### Table Formats

- **Results Summary**: Overview of all experiments
- **Detailed Comparison**: Side-by-side model comparison
- **Statistical Significance**: P-values and significance levels
- **Ablation Study**: Component contribution analysis

### Export Formats

- **PNG**: Raster images (150, 300, 600 DPI)
- **PDF**: Vector graphics for publications
- **SVG**: Scalable vector graphics
- **EPS**: Encapsulated PostScript

## Statistical Analysis

### Supported Tests

- **Paired t-test**: Compare two models
- **ANOVA**: Compare multiple models
- **Wilcoxon Signed-Rank Test**: Non-parametric comparison

### Effect Size Metrics

- **Cohen's d**: Standardized mean difference
- **Hedges' g**: Bias-corrected Cohen's d
- **Glass's delta**: Mean difference relative to control

## Advanced Features

### Mixed Precision Training

Enable Automatic Mixed Precision (AMP) for:
- Faster training (up to 2x speedup)
- Reduced memory usage
- Maintained accuracy

### Distributed Training

Multi-GPU training support for:
- Faster training on large datasets
- Handling larger batch sizes
- Scalable experiments

### Checkpointing

Automatic checkpoint saving:
- Save best model only
- Save optimizer state
- Save training state for resumption

### Early Stopping

Prevent overfitting:
- Monitor validation metrics
- Stop training when no improvement
- Restore best weights

## Troubleshooting

### Framework Not Found

If you see "Framework Not Found" error:
1. Check the framework path in `app.py`
2. Ensure the framework directory exists
3. Verify Python path includes the framework directory

### Model Loading Errors

If models fail to load:
1. Check model file exists in the framework
2. Verify model class name matches the file
3. Check for missing dependencies in the model file

### Dataset Loading Errors

If datasets fail to load:
1. Verify dataset path is correct
2. Check dataset format is supported
3. Ensure dataset has the required structure

### Visualization Issues

If visualizations don't render:
1. Check matplotlib backend is configured
2. Verify required packages are installed
3. Try clearing cache and reloading

## Performance Tips

### Training Speed

- Use mixed precision training
- Increase batch size if GPU memory allows
- Use multiple data loading workers
- Pin memory for faster data transfer

### Memory Usage

- Reduce batch size if out of memory
- Use gradient checkpointing for large models
- Clear cache between experiments
- Use smaller image sizes if possible

### Reproducibility

- Set random seed for all experiments
- Use deterministic algorithms
- Disable cuDNN benchmark for consistency
- Save and load configurations

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions, issues, or suggestions:
- GitHub Issues: https://github.com/your-repo/dl-experiment-manager/issues
- Email: your-email@example.com

## Acknowledgments

- Streamlit for the web framework
- PyTorch for the deep learning framework
- Plotly and Matplotlib for visualization
- The open-source community for various models and datasets

## Version History

### Version 1.0.0 (Current)
- Initial release
- Dynamic model and dataset loading
- Performance comparison with statistical tests
- Academic-quality visualization generation
- Configuration management
- Integration with PyTorch framework

## Roadmap

### Future Enhancements

- [ ] Real-time experiment monitoring
- [ ] Distributed experiment execution
- [ ] Cloud storage integration
- [ ] Collaborative experiment sharing
- [ ] AutoML integration
- [ ] Neural architecture search
- [ ] Hyperparameter optimization
- [ ] Experiment versioning and rollback

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{dl_experiment_manager,
  title={Deep Learning Experiment Manager},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/dl-experiment-manager}
}
```

## Appendix

### A. Supported Model Categories

1. **Classic Networks**: ResNet, VGG, AlexNet, Inception
2. **Attention Networks**: Transformer, Vision Transformer, Swin Transformer
3. **Lightweight Networks**: MobileNet, ShuffleNet, EfficientNet
4. **GAN Models**: DCGAN, StyleGAN, CycleGAN
5. **Object Detection**: YOLO, Faster R-CNN, SSD
6. **Semantic Segmentation**: UNet, DeepLab, FCN

### B. Supported Dataset Formats

1. **Image Folder**: root/class1/images, root/class2/images, ...
2. **CSV File**: With columns for image_path, label
3. **JSON File**: COCO format or custom JSON
4. **Custom Loader**: User-defined Python class

### C. Supported Evaluation Metrics

**Classification**:
- Accuracy, Precision, Recall, F1 Score
- Top-1 Accuracy, Top-5 Accuracy
- AUC-ROC, Confusion Matrix

**Detection**:
- mAP, mAP@50, mAP@75
- Average Recall, F1 Score per Class

**Segmentation**:
- IoU, Dice Coefficient
- Pixel Accuracy, Mean IoU
- Boundary IoU

**Generation**:
- FID, Inception Score
- LPIPS, SSIM

---

**Made with ❤️ for Research**