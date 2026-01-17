"""
Experiment Setup Page

This page allows users to configure training parameters,
select evaluation metrics, and set up experiments.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function for experiment setup page"""
    
    st.set_page_config(
        page_title="Experiment Setup - DL Experiment Manager",
        page_icon="⚙️",
        layout="wide"
    )
    
    # Title and description
    st.title("⚙️ Experiment Setup")
    st.markdown("""
    Configure training parameters, evaluation metrics, and experiment settings.
    Set up automated experiment runs with proper tracking.
    """)
    
    # Check if models and datasets are selected
    has_models = 'selected_models' in st.session_state and st.session_state.selected_models
    has_datasets = 'selected_datasets' in st.session_state and st.session_state.selected_datasets
    
    if not has_models or not has_datasets:
        st.warning("⚠️ Please select models and datasets before setting up experiments.")
        st.info("Go to Model Selection and Dataset Configuration pages first.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Model Selection", type="primary"):
                st.session_state.current_page = 'model_selection'
                st.rerun()
        with col2:
            if st.button("Go to Dataset Configuration", type="primary"):
                st.session_state.current_page = 'dataset_config'
                st.rerun()
        
        return
    
    # Main content in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Training Parameters", 
        "📊 Evaluation Metrics", 
        "⏱️ Experiment Schedule", 
        "🔧 Advanced Settings",
        "📋 Experiment Summary"
    ])
    
    with tab1:
        st.header("Training Parameters")
        
        # Basic training settings
        st.subheader("Basic Training Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=1000,
                value=100,
                help="Total number of training epochs"
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=512,
                value=32,
                help="Number of samples per batch"
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=1.0,
                value=0.001,
                format="%.5f",
                help="Initial learning rate for optimizer"
            )
        
        with col2:
            weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=0.1,
                value=0.0001,
                format="%.5f",
                help="L2 regularization parameter"
            )
            
            momentum = st.number_input(
                "Momentum",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                format="%.2f",
                help="Momentum for SGD optimizer"
            )
            
            gradient_clip = st.number_input(
                "Gradient Clipping",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                format="%.2f",
                help="Maximum gradient norm (0 to disable)"
            )
        
        # Optimizer selection
        st.subheader("Optimizer Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimizer_type = st.selectbox(
                "Optimizer",
                ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad"],
                index=0
            )
            
            if optimizer_type in ["Adam", "AdamW"]:
                beta1 = st.number_input("Beta1", min_value=0.0, max_value=1.0, value=0.9, format="%.3f")
                beta2 = st.number_input("Beta2", min_value=0.0, max_value=1.0, value=0.999, format="%.3f")
                epsilon = st.number_input("Epsilon", min_value=0.0, max_value=1.0, value=1e-8, format="%.1e")
        
        with col2:
            if optimizer_type == "SGD":
                nesterov = st.checkbox("Nesterov Momentum", value=True)
            
            amsgrad = st.checkbox("AMSGrad", value=False)
            weight_decay_type = st.selectbox(
                "Weight Decay Type",
                ["L2", "Decoupled", "None"],
                index=0
            )
        
        # Learning rate scheduler
        st.subheader("Learning Rate Scheduler")
        
        enable_scheduler = st.checkbox("Enable Learning Rate Scheduler", value=True)
        
        if enable_scheduler:
            scheduler_type = st.selectbox(
                "Scheduler Type",
                ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"],
                index=0
            )
            
            if scheduler_type == "StepLR":
                col1, col2 = st.columns(2)
                with col1:
                    step_size = st.number_input("Step Size", min_value=1, max_value=100, value=30)
                with col2:
                    gamma = st.number_input("Gamma", min_value=0.1, max_value=0.9, value=0.1, format="%.2f")
            
            elif scheduler_type == "CosineAnnealingLR":
                col1, col2 = st.columns(2)
                with col1:
                    T_max = st.number_input("T_max", min_value=1, max_value=1000, value=100)
                with col2:
                    eta_min = st.number_input("Eta_min", min_value=0.0, max_value=1.0, value=0.0, format="%.5f")
            
            elif scheduler_type == "ReduceLROnPlateau":
                col1, col2 = st.columns(2)
                with col1:
                    patience = st.number_input("Patience", min_value=1, max_value=50, value=10)
                    factor = st.number_input("Factor", min_value=0.1, max_value=0.9, value=0.5, format="%.2f")
                with col2:
                    threshold = st.number_input("Threshold", min_value=0.0001, max_value=1.0, value=0.0001, format="%.5f")
                    min_lr = st.number_input("Min LR", min_value=0.0, max_value=1.0, value=0.0, format="%.6f")
    
    with tab2:
        st.header("Evaluation Metrics")
        
        st.subheader("Primary Metrics")
        
        # Common metrics
        primary_metrics = st.multiselect(
            "Select Primary Metrics",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "AUC-ROC",
                "Top-1 Accuracy",
                "Top-5 Accuracy"
            ],
            default=["Accuracy", "F1 Score"]
        )
        
        # Task-specific metrics
        st.subheader("Task-Specific Metrics")
        
        task_type = st.selectbox(
            "Task Type",
            ["Classification", "Detection", "Segmentation", "Generation", "Regression"],
            index=0
        )
        
        if task_type == "Classification":
            classification_metrics = st.multiselect(
                "Classification Metrics",
                [
                    "Confusion Matrix",
                    "Classification Report",
                    "ROC Curve",
                    "Precision-Recall Curve"
                ],
                default=["Confusion Matrix"]
            )
        
        elif task_type == "Detection":
            detection_metrics = st.multiselect(
                "Detection Metrics",
                [
                    "mAP (mean Average Precision)",
                    "mAP@50",
                    "mAP@75",
                    "AR (Average Recall)",
                    "F1 Score per Class"
                ],
                default=["mAP", "mAP@50"]
            )
        
        elif task_type == "Segmentation":
            segmentation_metrics = st.multiselect(
                "Segmentation Metrics",
                [
                    "IoU (Intersection over Union)",
                    "Dice Coefficient",
                    "Pixel Accuracy",
                    "Mean IoU",
                    "Boundary IoU"
                ],
                default=["IoU", "Dice Coefficient"]
            )
        
        elif task_type == "Generation":
            generation_metrics = st.multiselect(
                "Generation Metrics",
                [
                    "FID (Fréchet Inception Distance)",
                    "IS (Inception Score)",
                    "LPIPS (Learned Perceptual Image Patch Similarity)",
                    "SSIM (Structural Similarity Index)"
                ],
                default=["FID", "IS"]
            )
        
        # Additional evaluation settings
        st.subheader("Evaluation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            eval_frequency = st.number_input(
                "Evaluation Frequency (epochs)",
                min_value=1,
                max_value=100,
                value=5,
                help="Evaluate model every N epochs"
            )
            
            save_best_only = st.checkbox("Save Best Model Only", value=True)
        
        with col2:
            early_stopping = st.checkbox("Enable Early Stopping", value=True)
            
            if early_stopping:
                patience = st.number_input(
                    "Early Stopping Patience",
                    min_value=1,
                    max_value=100,
                    value=20
                )
                monitor_metric = st.selectbox(
                    "Monitor Metric",
                    primary_metrics,
                    index=0
                )
                mode = st.selectbox(
                    "Mode",
                    ["min", "max"],
                    index=1 if monitor_metric in ["Accuracy", "F1 Score", "AUC-ROC"] else 0
                )
    
    with tab3:
        st.header("Experiment Schedule")
        
        st.subheader("Experiment Configuration")
        
        # Selected models and datasets
        st.write("### Selected Models")
        for i, model in enumerate(st.session_state.selected_models):
            st.write(f"{i+1}. {model}")
        
        st.write("### Selected Datasets")
        for i, dataset in enumerate(st.session_state.selected_datasets):
            st.write(f"{i+1}. {dataset['name']} ({dataset['category']})")
        
        # Experiment combinations
        st.subheader("Experiment Combinations")
        
        experiment_mode = st.radio(
            "Experiment Mode",
            ["All Combinations", "Manual Selection", "Sequential"],
            horizontal=True
        )
        
        if experiment_mode == "All Combinations":
            num_experiments = len(st.session_state.selected_models) * len(st.session_state.selected_datasets)
            st.info(f"Will create {num_experiments} experiment combinations")
            
            if st.button("Generate All Combinations", type="primary"):
                experiments = []
                for model in st.session_state.selected_models:
                    for dataset in st.session_state.selected_datasets:
                        experiments.append({
                            'model': model,
                            'dataset': dataset['name'],
                            'status': 'pending'
                        })
                st.session_state.experiments = experiments
                st.success(f"Generated {len(experiments)} experiments!")
        
        elif experiment_mode == "Manual Selection":
            st.write("Select specific model-dataset combinations:")
            
            experiments_to_run = []
            for model in st.session_state.selected_models:
                st.write(f"**{model}**")
                for dataset in st.session_state.selected_datasets:
                    if st.checkbox(f"{model} + {dataset['name']}", key=f"exp_{model}_{dataset['name']}"):
                        experiments_to_run.append({
                            'model': model,
                            'dataset': dataset['name'],
                            'status': 'pending'
                        })
            
            if st.button("Create Selected Experiments", type="primary"):
                st.session_state.experiments = experiments_to_run
                st.success(f"Created {len(experiments_to_run)} experiments!")
        
        elif experiment_mode == "Sequential":
            st.info("Run experiments one by one, with manual approval between runs")
            st.warning("This mode requires manual intervention during execution")
        
        # Execution settings
        st.subheader("Execution Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            device = st.selectbox(
                "Device",
                ["Auto (CUDA if available)", "CPU", "CUDA:0", "CUDA:1"],
                index=0
            )
            
            num_workers = st.number_input(
                "Data Loading Workers",
                min_value=0,
                max_value=16,
                value=4
            )
        
        with col2:
            save_frequency = st.number_input(
                "Checkpoint Frequency (epochs)",
                min_value=1,
                max_value=100,
                value=10
            )
            
            log_frequency = st.number_input(
                "Log Frequency (batches)",
                min_value=1,
                max_value=1000,
                value=100
            )
    
    with tab4:
        st.header("Advanced Settings")
        
        st.subheader("Checkpointing and Logging")
        
        col1, col2 = st.columns(2)
        
        with col1:
            checkpoint_dir = st.text_input(
                "Checkpoint Directory",
                value="./checkpoints",
                help="Directory to save model checkpoints"
            )
            
            log_dir = st.text_input(
                "Log Directory",
                value="./logs",
                help="Directory to save training logs"
            )
        
        with col2:
            save_optimizer_state = st.checkbox("Save Optimizer State", value=True)
            save_training_state = st.checkbox("Save Training State", value=True)
            enable_tensorboard = st.checkbox("Enable TensorBoard Logging", value=True)
        
        # Mixed precision and distributed training
        st.subheader("Training Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_amp = st.checkbox("Use Automatic Mixed Precision (AMP)", value=False)
            use_distributed = st.checkbox("Enable Distributed Training", value=False)
        
        with col2:
            if use_amp:
                amp_backend = st.selectbox(
                    "AMP Backend",
                    ["native", "apex"],
                    index=0
                )
            
            if use_distributed:
                num_gpus = st.number_input(
                    "Number of GPUs",
                    min_value=1,
                    max_value=8,
                    value=2
                )
        
        # Reproducibility
        st.subheader("Reproducibility")
        
        col1, col2 = st.columns(2)
        
        with col1:
            set_seed = st.checkbox("Set Random Seed", value=True)
            
            if set_seed:
                random_seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=2147483647,
                    value=42
                )
        
        with col2:
            deterministic = st.checkbox("Deterministic Algorithms", value=False)
            benchmark = st.checkbox("Enable cuDNN Benchmark", value=False)
        
        # Custom hooks and callbacks
        st.subheader("Custom Hooks")
        
        enable_custom_hooks = st.checkbox("Enable Custom Hooks", value=False)
        
        if enable_custom_hooks:
            st.text_area(
                "Custom Hook Code",
                placeholder="# Define your custom hooks here\n# Example:\ndef on_epoch_end(epoch, metrics):\n    print(f'Epoch {epoch} completed')",
                height=200
            )
    
    with tab5:
        st.header("Experiment Summary")
        
        st.subheader("Configuration Summary")
        
        # Training parameters summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Training Parameters")
            st.write(f"- **Epochs:** {epochs}")
            st.write(f"- **Batch Size:** {batch_size}")
            st.write(f"- **Learning Rate:** {learning_rate}")
            st.write(f"- **Optimizer:** {optimizer_type}")
            st.write(f"- **Scheduler:** {scheduler_type if enable_scheduler else 'None'}")
        
        with col2:
            st.write("### Evaluation Settings")
            st.write(f"- **Primary Metrics:** {', '.join(primary_metrics)}")
            st.write(f"- **Task Type:** {task_type}")
            st.write(f"- **Early Stopping:** {'Enabled' if early_stopping else 'Disabled'}")
            st.write(f"- **Eval Frequency:** Every {eval_frequency} epochs")
        
        # Experiments to run
        st.subheader("Experiments to Run")
        
        if 'experiments' in st.session_state and st.session_state.experiments:
            st.write(f"**Total Experiments:** {len(st.session_state.experiments)}")
            
            import pandas as pd
            exp_data = []
            for i, exp in enumerate(st.session_state.experiments):
                exp_data.append({
                    'ID': i+1,
                    'Model': exp['model'],
                    'Dataset': exp['dataset'],
                    'Status': exp['status']
                })
            
            df = pd.DataFrame(exp_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No experiments configured yet. Go to 'Experiment Schedule' tab to create experiments.")
        
        # Estimated time and resources
        st.subheader("Resource Estimation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estimated Time", "~24 hours")
        with col2:
            st.metric("GPU Memory", "~8 GB")
        with col3:
            st.metric("Disk Space", "~50 GB")
        
        # Actions
        st.subheader("Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save Configuration", type="primary"):
                config = {
                    'training': {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'optimizer': optimizer_type,
                        'scheduler': scheduler_type if enable_scheduler else None
                    },
                    'evaluation': {
                        'primary_metrics': primary_metrics,
                        'task_type': task_type,
                        'early_stopping': early_stopping
                    },
                    'experiments': st.session_state.get('experiments', [])
                }
                st.session_state.experiment_config = config
                st.success("Configuration saved!")
        
        with col2:
            if st.button("Export as Python Script"):
                st.info("Python script export would be implemented here.")
        
        with col3:
            if st.button("Start Experiments"):
                st.info("Experiments would be started here.")
    
    # Sidebar
    with st.sidebar:
        st.header("Experiment Setup Help")
        
        st.markdown("""
        ### How to Use This Page
        
        1. **Training Parameters**: Configure optimizer, learning rate, etc.
        2. **Evaluation Metrics**: Select metrics to track
        3. **Schedule**: Create experiment combinations
        4. **Advanced**: Set up checkpointing, logging, etc.
        5. **Summary**: Review before starting
        
        ### Tips
        
        - Use learning rate scheduling for better convergence
        - Enable early stopping to prevent overfitting
        - Save checkpoints regularly to avoid data loss
        - Use mixed precision for faster training
        - Set random seed for reproducibility
        """)
        
        st.markdown("---")
        
        # Quick actions
        st.header("Quick Actions")
        
        if st.button("Load Template"):
            st.info("Template loading would be implemented here.")
        
        if st.button("Reset to Defaults"):
            st.info("Reset to default settings.")

if __name__ == "__main__":
    main()