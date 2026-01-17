"""
Dataset Configuration Page

This page allows users to select, configure, and prepare datasets
for deep learning experiments.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function for dataset configuration page"""
    
    st.set_page_config(
        page_title="Dataset Configuration - DL Experiment Manager",
        page_icon="📊",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 Dataset Configuration")
    st.markdown("""
    Select and configure datasets for your experiments.
    Support for built-in datasets and custom data loading.
    """)
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Select Dataset", 
        "⚙️ Configure Dataset", 
        "📊 Data Preview", 
        "📋 Selected Datasets"
    ])
    
    with tab1:
        st.header("Select Dataset")
        
        # Dataset source selection
        dataset_source = st.radio(
            "Dataset Source",
            ["Built-in Datasets", "Custom Dataset", "Upload Dataset"],
            horizontal=True
        )
        
        if dataset_source == "Built-in Datasets":
            st.subheader("Built-in Datasets")
            
            # Dataset categories
            dataset_categories = {
                "Image Classification": ["CIFAR10", "CIFAR100", "MNIST", "ImageNet", "FashionMNIST"],
                "Object Detection": ["COCO", "Pascal VOC", "Cityscapes"],
                "Semantic Segmentation": ["Pascal VOC Segmentation", "Cityscapes", "ADE20K"],
                "Text Classification": ["IMDB", "AG News", "SST-2"],
                "Machine Translation": ["WMT14", "IWSLT"],
                "Graph Data": ["Cora", "CiteSeer", "PubMed"]
            }
            
            selected_category = st.selectbox(
                "Select Dataset Category",
                list(dataset_categories.keys()),
                index=0
            )
            
            if selected_category:
                datasets = dataset_categories[selected_category]
                selected_dataset = st.selectbox(
                    "Select Dataset",
                    datasets,
                    index=0
                )
                
                if selected_dataset:
                    # Dataset information
                    with st.expander("Dataset Information", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Name:** {selected_dataset}")
                            st.write(f"**Category:** {selected_category}")
                            st.write(f"**Size:** To be determined")
                        
                        with col2:
                            st.write(f"**Classes:** To be determined")
                            st.write(f"**Format:** To be determined")
                            st.write(f"**License:** To be determined")
                    
                    # Quick actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Add to Selection", type="primary"):
                            if 'selected_datasets' not in st.session_state:
                                st.session_state.selected_datasets = []
                            
                            dataset_info = {
                                'name': selected_dataset,
                                'category': selected_category,
                                'source': 'builtin'
                            }
                            
                            if dataset_info not in st.session_state.selected_datasets:
                                st.session_state.selected_datasets.append(dataset_info)
                                st.success(f"Added {selected_dataset} to selection")
                            else:
                                st.warning(f"{selected_dataset} is already selected")
                    
                    with col2:
                        if st.button("View Details"):
                            st.session_state.dataset_details = selected_dataset
                            st.rerun()
        
        elif dataset_source == "Custom Dataset":
            st.subheader("Custom Dataset")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_path = st.text_input(
                    "Dataset Path",
                    placeholder="E:/path/to/your/dataset",
                    help="Full path to your dataset directory"
                )
                
                dataset_format = st.selectbox(
                    "Dataset Format",
                    ["Image Folder", "CSV File", "JSON File", "Custom Loader"],
                    index=0
                )
            
            with col2:
                dataset_name = st.text_input(
                    "Dataset Name",
                    placeholder="My Custom Dataset"
                )
                
                dataset_type = st.selectbox(
                    "Dataset Type",
                    ["Classification", "Detection", "Segmentation", "Other"],
                    index=0
                )
            
            if dataset_format == "Image Folder":
                st.info("Expected structure: root/class1/images, root/class2/images, ...")
            
            if st.button("Load Custom Dataset", type="primary"):
                if dataset_path and dataset_name:
                    if 'selected_datasets' not in st.session_state:
                        st.session_state.selected_datasets = []
                    
                    dataset_info = {
                        'name': dataset_name,
                        'category': dataset_type,
                        'source': 'custom',
                        'path': dataset_path,
                        'format': dataset_format
                    }
                    
                    st.session_state.selected_datasets.append(dataset_info)
                    st.success(f"Loaded custom dataset: {dataset_name}")
                else:
                    st.warning("Please provide both dataset path and name")
        
        elif dataset_source == "Upload Dataset":
            st.subheader("Upload Dataset")
            
            st.info("Upload your dataset files directly to the application.")
            
            uploaded_file = st.file_uploader(
                "Upload Dataset File",
                type=['zip', 'tar', 'csv', 'json'],
                help="Supported formats: ZIP, TAR, CSV, JSON"
            )
            
            if uploaded_file:
                st.success(f"File uploaded: {uploaded_file.name}")
                st.info("Dataset will be extracted and processed in the background.")
    
    with tab2:
        st.header("Configure Dataset Parameters")
        
        # General configuration
        st.subheader("General Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=512,
                value=32,
                help="Number of samples per batch"
            )
            
            num_workers = st.number_input(
                "Number of Workers",
                min_value=0,
                max_value=16,
                value=4,
                help="Number of subprocesses for data loading"
            )
        
        with col2:
            shuffle = st.checkbox("Shuffle Data", value=True)
            drop_last = st.checkbox("Drop Last Batch", value=False)
            pin_memory = st.checkbox("Pin Memory", value=True)
        
        # Data splitting
        st.subheader("Data Splitting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_split = st.slider(
                "Training Split",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Percentage of data for training"
            )
        
        with col2:
            val_split = st.slider(
                "Validation Split",
                min_value=0.05,
                max_value=0.3,
                value=0.1,
                step=0.05,
                help="Percentage of data for validation"
            )
        
        with col3:
            test_split = st.slider(
                "Test Split",
                min_value=0.05,
                max_value=0.2,
                value=0.1,
                step=0.05,
                help="Percentage of data for testing"
            )
        
        # Validate splits
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 0.01:
            st.error(f"Splits must sum to 1.0 (current: {total_split:.2f})")
        else:
            st.success(f"Splits are valid: Train {train_split:.0%}, Val {val_split:.0%}, Test {test_split:.0%}")
        
        # Data augmentation
        st.subheader("Data Augmentation")
        
        enable_augmentation = st.checkbox("Enable Data Augmentation", value=True)
        
        if enable_augmentation:
            col1, col2 = st.columns(2)
            
            with col1:
                horizontal_flip = st.checkbox("Horizontal Flip", value=True)
                vertical_flip = st.checkbox("Vertical Flip", value=False)
                rotation = st.slider("Rotation Range", min_value=0, max_value=180, value=15)
            
            with col2:
                brightness = st.slider("Brightness", min_value=0.5, max_value=1.5, value=1.0)
                contrast = st.slider("Contrast", min_value=0.5, max_value=1.5, value=1.0)
                normalize = st.checkbox("Normalize", value=True)
        
        # Save configuration
        if st.button("Save Dataset Configuration", type="primary"):
            config = {
                'batch_size': batch_size,
                'num_workers': num_workers,
                'shuffle': shuffle,
                'drop_last': drop_last,
                'pin_memory': pin_memory,
                'train_split': train_split,
                'val_split': val_split,
                'test_split': test_split,
                'augmentation': {
                    'enabled': enable_augmentation,
                    'horizontal_flip': horizontal_flip if enable_augmentation else False,
                    'vertical_flip': vertical_flip if enable_augmentation else False,
                    'rotation': rotation if enable_augmentation else 0,
                    'brightness': brightness if enable_augmentation else 1.0,
                    'contrast': contrast if enable_augmentation else 1.0,
                    'normalize': normalize if enable_augmentation else False
                }
            }
            st.session_state.dataset_config = config
            st.success("Dataset configuration saved!")
    
    with tab3:
        st.header("Data Preview and Statistics")
        
        if 'selected_datasets' in st.session_state and st.session_state.selected_datasets:
            st.subheader("Selected Datasets Preview")
            
            for i, dataset in enumerate(st.session_state.selected_datasets):
                with st.expander(f"Dataset {i+1}: {dataset['name']}", expanded=i==0):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {dataset['name']}")
                        st.write(f"**Category:** {dataset['category']}")
                        st.write(f"**Source:** {dataset.get('source', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Path:** {dataset.get('path', 'N/A')}")
                        st.write(f"**Format:** {dataset.get('format', 'N/A')}")
            
            # Statistics
            st.subheader("Dataset Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", "50,000")
            with col2:
                st.metric("Training Samples", "40,000")
            with col3:
                st.metric("Validation Samples", "5,000")
            with col4:
                st.metric("Test Samples", "5,000")
            
            # Class distribution
            st.subheader("Class Distribution")
            
            import pandas as pd
            import plotly.express as px
            
            # Example class distribution
            class_data = {
                'Class': list(range(10)),
                'Count': [5000] * 10
            }
            df = pd.DataFrame(class_data)
            
            fig = px.bar(df, x='Class', y='Count', title='Class Distribution')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datasets selected yet. Select a dataset from the 'Select Dataset' tab.")
    
    with tab4:
        st.header("Selected Datasets")
        
        if 'selected_datasets' in st.session_state and st.session_state.selected_datasets:
            st.write(f"**Total Selected:** {len(st.session_state.selected_datasets)} datasets")
            
            for i, dataset in enumerate(st.session_state.selected_datasets):
                with st.expander(f"Dataset {i+1}: {dataset['name']}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Name:** {dataset['name']}")
                        st.write(f"**Category:** {dataset['category']}")
                        st.write(f"**Source:** {dataset.get('source', 'N/A')}")
                        if 'path' in dataset:
                            st.write(f"**Path:** {dataset['path']}")
                    
                    with col2:
                        if st.button(f"Remove", key=f"remove_dataset_{i}"):
                            st.session_state.selected_datasets.pop(i)
                            st.rerun()
            
            # Export options
            st.subheader("Export Dataset Configuration")
            
            if st.button("Export as JSON", type="primary"):
                import json
                config = {
                    'datasets': st.session_state.selected_datasets,
                    'config': st.session_state.get('dataset_config', {})
                }
                st.download_button(
                    label="Download Configuration",
                    data=json.dumps(config, indent=4),
                    file_name="dataset_config.json",
                    mime="application/json"
                )
        else:
            st.info("No datasets selected yet. Browse datasets and add them to your selection.")
    
    # Sidebar
    with st.sidebar:
        st.header("Dataset Configuration Help")
        
        st.markdown("""
        ### How to Use This Page
        
        1. **Select Dataset**: Choose from built-in datasets or use custom data
        2. **Configure**: Adjust data loading and augmentation parameters
        3. **Preview**: View dataset statistics and class distribution
        4. **Select**: Add datasets to your selection for experiments
        
        ### Tips
        
        - Use data augmentation to improve model generalization
        - Adjust batch size based on available GPU memory
        - Ensure proper train/val/test splits
        - Check class balance for classification tasks
        """)
        
        st.markdown("---")
        
        # Quick actions
        st.header("Quick Actions")
        
        if st.button("Clear All Selections"):
            if 'selected_datasets' in st.session_state:
                st.session_state.selected_datasets = []
                st.rerun()
        
        if st.button("Load Template"):
            st.info("Template loading would be implemented here.")

if __name__ == "__main__":
    main()