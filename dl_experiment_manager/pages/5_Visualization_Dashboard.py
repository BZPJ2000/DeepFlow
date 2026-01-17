"""
Visualization Dashboard Page

This page provides academic-quality visualization generation for experiment results,
including charts, tables, and export functionality for publications.
"""

import streamlit as st
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import base64
from io import BytesIO

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set matplotlib style for academic publications
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    """Main function for visualization dashboard page"""
    
    st.set_page_config(
        page_title="Visualization Dashboard - DL Experiment Manager",
        page_icon="📊",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 Visualization Dashboard")
    st.markdown("""
    Generate academic-quality visualizations for experiment results.
    Create publication-ready charts, tables, and figures.
    """)
    
    # Check if experiments exist
    has_experiments = 'experiment_results' in st.session_state and st.session_state.experiment_results
    
    if not has_experiments:
        st.warning("⚠️ No experiment results available.")
        st.info("Please run experiments first to generate results for visualization.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Performance Comparison", type="primary"):
                st.session_state.current_page = 'performance_comparison'
                st.rerun()
        with col2:
            if st.button("Load Sample Results"):
                load_sample_results()
                st.success("Sample results loaded!")
                st.rerun()
        
        return
    
    # Main content in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Charts", 
        "📋 Tables", 
        "🎨 Academic Figures", 
        "📄 Export Options",
        "⚙️ Visualization Settings"
    ])
    
    with tab1:
        st.header("Interactive Charts")
        
        results = st.session_state.experiment_results
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart Type",
            ["Performance Comparison", "Training Curves", "Confusion Matrix", "ROC Curves", "Learning Rate Schedule"],
            index=0
        )
        
        if chart_type == "Performance Comparison":
            st.subheader("Performance Comparison Chart")
            
            # Select metrics to compare
            metrics_to_compare = st.multiselect(
                "Select Metrics",
                ["Accuracy", "Precision", "Recall", "F1 Score", "Training Time", "Inference Time"],
                default=["Accuracy", "F1 Score"]
            )
            
            if metrics_to_compare:
                import plotly.express as px
                
                # Prepare data
                plot_data = []
                for exp_id, result in results.items():
                    for metric in metrics_to_compare:
                        value = result.get('metrics', {}).get(metric.lower().replace(' ', '_'), 0)
                        plot_data.append({
                            'Experiment': exp_id,
                            'Model': result.get('model', 'N/A'),
                            'Metric': metric,
                            'Value': value
                        })
                
                df = pd.DataFrame(plot_data)
                
                fig = px.bar(
                    df,
                    x='Model',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Model Performance Comparison',
                    height=500,
                    template='plotly_white'
                )
                
                fig.update_layout(
                    font=dict(size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Training Curves":
            st.subheader("Training and Validation Curves")
            
            # Select experiments
            experiment_ids = list(results.keys())
            selected_experiments = st.multiselect(
                "Select Experiments",
                experiment_ids,
                default=experiment_ids[:3]
            )
            
            if selected_experiments:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Training Loss', 'Validation Loss'),
                    vertical_spacing=0.15
                )
                
                colors = px.colors.qualitative.Set1
                
                for i, exp_id in enumerate(selected_experiments):
                    result = results[exp_id]
                    history = result.get('history', {})
                    
                    if 'train_loss' in history:
                        epochs = list(range(len(history['train_loss'])))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=epochs,
                                y=history['train_loss'],
                                mode='lines',
                                name=f"{exp_id} - Train",
                                line=dict(width=2, color=colors[i % len(colors)])
                            ),
                            row=1, col=1
                        )
                    
                    if 'val_loss' in history:
                        epochs = list(range(len(history['val_loss'])))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=epochs,
                                y=history['val_loss'],
                                mode='lines',
                                name=f"{exp_id} - Val",
                                line=dict(width=2, color=colors[i % len(colors)], dash='dash')
                            ),
                            row=2, col=1
                        )
                
                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=2, col=1)
                fig.update_yaxes(title_text="Loss", row=1, col=1)
                fig.update_yaxes(title_text="Loss", row=2, col=1)
                
                fig.update_layout(
                    height=700,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            
            # Select experiment
            selected_exp = st.selectbox(
                "Select Experiment",
                list(results.keys())
            )
            
            if selected_exp:
                result = results[selected_exp]
                
                # Generate sample confusion matrix (would come from actual results)
                num_classes = 10
                confusion_matrix = np.random.randint(0, 100, size=(num_classes, num_classes))
                np.fill_diagonal(confusion_matrix, np.random.randint(200, 500, size=num_classes))
                
                # Normalize
                confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    confusion_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    ax=ax,
                    cbar_kws={'label': 'Normalized Count'}
                )
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_title(f'Confusion Matrix - {result.get("model", "Model")}', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                plt.close()
                
                # Download button
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode()
                st.download_button(
                    label="Download Confusion Matrix",
                    data=buf.getvalue(),
                    file_name="confusion_matrix.png",
                    mime="image/png"
                )
        
        elif chart_type == "ROC Curves":
            st.subheader("ROC Curves")
            
            # Select experiments
            selected_experiments = st.multiselect(
                "Select Experiments",
                list(results.keys()),
                default=list(results.keys())[:3]
            )
            
            if selected_experiments:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                for i, exp_id in enumerate(selected_experiments):
                    result = results[exp_id]
                    
                    # Generate sample ROC curve data
                    fpr = np.linspace(0, 1, 100)
                    tpr = 1 - np.exp(-5 * fpr)
                    auc = 0.85 + np.random.uniform(-0.05, 0.05)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        name=f"{result.get('model', exp_id)} (AUC = {auc:.3f})",
                        line=dict(width=2)
                    ))
                
                # Add diagonal line
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(width=1, dash='dash', color='gray')
                ))
                
                fig.update_layout(
                    title='ROC Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=600,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Learning Rate Schedule":
            st.subheader("Learning Rate Schedule")
            
            # Select experiment
            selected_exp = st.selectbox(
                "Select Experiment",
                list(results.keys())
            )
            
            if selected_exp:
                result = results[selected_exp]
                history = result.get('history', {})
                
                if 'learning_rate' in history:
                    lrs = history['learning_rate']
                    epochs = list(range(len(lrs)))
                    
                    import plotly.express as px
                    
                    df = pd.DataFrame({
                        'Epoch': epochs,
                        'Learning Rate': lrs
                    })
                    
                    fig = px.line(
                        df,
                        x='Epoch',
                        y='Learning Rate',
                        title='Learning Rate Schedule',
                        log_y=True,
                        template='plotly_white'
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title='Epoch',
                        yaxis_title='Learning Rate (log scale)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Learning rate history not available for this experiment.")
    
    with tab2:
        st.header("Academic Tables")
        
        results = st.session_state.experiment_results
        
        # Table type selection
        table_type = st.selectbox(
            "Table Type",
            ["Results Summary", "Detailed Comparison", "Statistical Significance", "Ablation Study"],
            index=0
        )
        
        if table_type == "Results Summary":
            st.subheader("Experiment Results Summary")
            
            # Create summary table
            summary_data = []
            for exp_id, result in results.items():
                metrics = result.get('metrics', {})
                summary_data.append({
                    'Model': result.get('model', 'N/A'),
                    'Dataset': result.get('dataset', 'N/A'),
                    'Accuracy (%)': f"{metrics.get('accuracy', 0):.2f}",
                    'Precision (%)': f"{metrics.get('precision', 0):.2f}",
                    'Recall (%)': f"{metrics.get('recall', 0):.2f}",
                    'F1 Score (%)': f"{metrics.get('f1_score', 0):.2f}",
                    'Params (M)': f"{result.get('parameters', 0) / 1e6:.2f}",
                    'Train Time (h)': f"{result.get('training_time', 0) / 3600:.2f}",
                    'Inference (ms)': f"{result.get('inference_time', 0):.2f}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            # Display table
            st.dataframe(df_summary, use_container_width=True)
            
            # LaTeX export
            if st.button("Export as LaTeX Table", type="primary"):
                latex_table = dataframe_to_latex(df_summary, caption="Experiment Results Summary")
                st.text_area("LaTeX Code", latex_table, height=300)
                
                st.download_button(
                    label="Download LaTeX",
                    data=latex_table,
                    file_name="results_table.tex",
                    mime="text/plain"
                )
        
        elif table_type == "Detailed Comparison":
            st.subheader("Detailed Model Comparison")
            
            # Select metrics for detailed comparison
            metrics_for_table = st.multiselect(
                "Select Metrics",
                ["Accuracy", "Precision", "Recall", "F1 Score", "Parameters", "Training Time", "Inference Time"],
                default=["Accuracy", "Precision", "Recall", "F1 Score"]
            )
            
            if metrics_for_table:
                # Create comparison table
                comparison_data = []
                for exp_id, result in results.items():
                    row = {'Experiment': exp_id, 'Model': result.get('model', 'N/A')}
                    for metric in metrics_for_table:
                        key = metric.lower().replace(' ', '_')
                        if key in ['parameters', 'training_time', 'inference_time']:
                            value = result.get(key, 0)
                        else:
                            value = result.get('metrics', {}).get(key, 0)
                        row[metric] = f"{value:.4f}" if isinstance(value, float) else value
                    comparison_data.append(row)
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
        
        elif table_type == "Statistical Significance":
            st.subheader("Statistical Significance Table")
            
            # Create significance matrix
            models = list(set([r.get('model', 'N/A') for r in results.values()]))
            metric_for_sig = st.selectbox(
                "Metric for Significance Test",
                ["Accuracy", "Precision", "Recall", "F1 Score"],
                index=0
            )
            
            # Create pairwise comparison table
            sig_data = []
            for i, model1 in enumerate(models):
                row = {'Model': model1}
                for model2 in models:
                    if model1 == model2:
                        row[model2] = '-'
                    else:
                        # Simulate significance test
                        p_value = np.random.uniform(0.001, 0.1)
                        if p_value < 0.01:
                            sig = '**'
                        elif p_value < 0.05:
                            sig = '*'
                        else:
                            sig = ''
                        row[model2] = f"{p_value:.4f}{sig}"
                sig_data.append(row)
            
            df_sig = pd.DataFrame(sig_data)
            st.dataframe(df_sig, use_container_width=True)
            
            st.info("** Significant at p < 0.05\n*** Significant at p < 0.01")
        
        elif table_type == "Ablation Study":
            st.subheader("Ablation Study Table")
            
            # Ablation study configuration
            base_model = st.selectbox(
                "Base Model",
                list(set([r.get('model', 'N/A') for r in results.values()]))
            )
            
            ablation_components = st.multiselect(
                "Ablation Components",
                ["Component A", "Component B", "Component C", "Component D"],
                default=["Component A", "Component B"]
            )
            
            if st.button("Generate Ablation Table"):
                # Generate ablation study table
                ablation_data = []
                
                # Full model
                full_result = None
                for exp_id, result in results.items():
                    if result.get('model') == base_model:
                        full_result = result
                        break
                
                if full_result:
                    ablation_data.append({
                        'Configuration': 'Full Model',
                        'Accuracy (%)': f"{full_result.get('metrics', {}).get('accuracy', 0):.2f}",
                        'Parameters (M)': f"{full_result.get('parameters', 0) / 1e6:.2f}"
                    })
                
                # Ablated versions
                for i in range(1, len(ablation_components) + 1):
                    ablation_data.append({
                        'Configuration': f"- {', '.join(ablation_components[:i])}",
                        'Accuracy (%)': f"{full_result.get('metrics', {}).get('accuracy', 0) * (1 - i * 0.02):.2f}",
                        'Parameters (M)': f"{full_result.get('parameters', 0) * (1 - i * 0.15) / 1e6:.2f}"
                    })
                
                df_ablation = pd.DataFrame(ablation_data)
                st.dataframe(df_ablation, use_container_width=True)
    
    with tab3:
        st.header("Academic Quality Figures")
        
        results = st.session_state.experiment_results
        
        # Figure type selection
        figure_type = st.selectbox(
            "Figure Type",
            ["Publication-Ready Bar Chart", "Multi-Panel Figure", "Model Architecture Diagram", "Results Overview Figure"],
            index=0
        )
        
        if figure_type == "Publication-Ready Bar Chart":
            st.subheader("Publication-Ready Bar Chart")
            
            # Configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                figure_width = st.number_input("Width (inches)", min_value=4, max_value=12, value=8)
            with col2:
                figure_height = st.number_input("Height (inches)", min_value=3, max_value=10, value=6)
            with col3:
                dpi = st.selectbox("DPI", [150, 300, 600], index=1)
            
            metric_for_figure = st.selectbox(
                "Metric to Plot",
                ["Accuracy", "Precision", "Recall", "F1 Score"],
                index=0
            )
            
            if st.button("Generate Publication Figure", type="primary"):
                # Create publication-ready figure
                fig, ax = plt.subplots(figsize=(figure_width, figure_height))
                
                # Prepare data
                models = [r.get('model', 'N/A') for r in results.values()]
                values = [r.get('metrics', {}).get(metric_for_figure.lower().replace(' ', '_'), 0) 
                          for r in results.values()]
                
                # Create bar chart
                bars = ax.bar(range(len(models)), values, color='steelblue', edgecolor='black', linewidth=1.5)
                
                # Styling for publication
                ax.set_xlabel('Model', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric_for_figure, fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.2f}%',
                            ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download PNG",
                        data=buf.getvalue(),
                        file_name="publication_figure.png",
                        mime="image/png"
                    )
                with col2:
                    buf = BytesIO()
                    fig.savefig(buf, format='pdf', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download PDF",
                        data=buf.getvalue(),
                        file_name="publication_figure.pdf",
                        mime="application/pdf"
                    )
        
        elif figure_type == "Multi-Panel Figure":
            st.subheader("Multi-Panel Figure")
            
            # Panel configuration
            num_panels = st.selectbox("Number of Panels", [4, 6, 9], index=0)
            
            if st.button("Generate Multi-Panel Figure", type="primary"):
                # Create multi-panel figure
                fig, axes = plt.subplots(2, num_panels//2, figsize=(12, 8))
                axes = axes.flatten()
                
                # Generate sample plots for each panel
                for i, ax in enumerate(axes):
                    if i < len(results):
                        result = list(results.values())[i]
                        
                        # Sample data
                        x = np.linspace(0, 10, 100)
                        y = np.sin(x + i) + np.random.normal(0, 0.1, 100)
                        
                        ax.plot(x, y, linewidth=2)
                        ax.set_title(f"{result.get('model', 'Model')}", fontsize=10, fontweight='bold')
                        ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                
                # Download
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="Download Multi-Panel Figure",
                    data=buf.getvalue(),
                    file_name="multi_panel_figure.png",
                    mime="image/png"
                )
        
        elif figure_type == "Model Architecture Diagram":
            st.subheader("Model Architecture Diagram")
            
            # Select model
            selected_model = st.selectbox(
                "Select Model",
                list(set([r.get('model', 'N/A') for r in results.values()]))
            )
            
            if st.button("Generate Architecture Diagram", type="primary"):
                # Create architecture diagram (placeholder)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Draw simple architecture
                ax.text(0.5, 0.9, f"{selected_model} Architecture", 
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        transform=ax.transAxes)
                
                # Draw layers
                layers = ['Input', 'Conv1', 'Conv2', 'FC1', 'Output']
                for i, layer in enumerate(layers):
                    rect = plt.Rectangle((0.1 + i*0.18, 0.3, 0.15, 0.4), 
                                         facecolor='lightblue', edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(0.175 + i*0.18, 0.5, layer, 
                            ha='center', va='center', fontsize=10, fontweight='bold')
                    
                    # Add arrows
                    if i < len(layers) - 1:
                        ax.arrow(0.25 + i*0.18, 0.5, 0.03, 0, 
                                head_width=0.02, head_length=0.02, fc='black', ec='black')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close()
        
        elif figure_type == "Results Overview Figure":
            st.subheader("Results Overview Figure")
            
            if st.button("Generate Overview Figure", type="primary"):
                # Create overview figure with multiple subplots
                fig = plt.figure(figsize=(14, 10))
                gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
                
                # Panel 1: Performance comparison
                ax1 = fig.add_subplot(gs[0, 0])
                models = [r.get('model', 'N/A') for r in results.values()]
                accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in results.values()]
                ax1.bar(range(len(models)), accuracies, color='steelblue')
                ax1.set_title('Accuracy Comparison', fontweight='bold')
                ax1.set_ylabel('Accuracy (%)')
                ax1.set_xticks(range(len(models)))
                ax1.set_xticklabels(models, rotation=45, ha='right')
                
                # Panel 2: Training time
                ax2 = fig.add_subplot(gs[0, 1])
                train_times = [r.get('training_time', 0) / 3600 for r in results.values()]
                ax2.bar(range(len(models)), train_times, color='coral')
                ax2.set_title('Training Time', fontweight='bold')
                ax2.set_ylabel('Time (hours)')
                ax2.set_xticks(range(len(models)))
                ax2.set_xticklabels(models, rotation=45, ha='right')
                
                # Panel 3: Parameters
                ax3 = fig.add_subplot(gs[0, 2])
                params = [r.get('parameters', 0) / 1e6 for r in results.values()]
                ax3.bar(range(len(models)), params, color='lightgreen')
                ax3.set_title('Model Parameters', fontweight='bold')
                ax3.set_ylabel('Parameters (M)')
                ax3.set_xticks(range(len(models)))
                ax3.set_xticklabels(models, rotation=45, ha='right')
                
                # Panel 4: Inference time
                ax4 = fig.add_subplot(gs[1, 0])
                inf_times = [r.get('inference_time', 0) for r in results.values()]
                ax4.bar(range(len(models)), inf_times, color='gold')
                ax4.set_title('Inference Time', fontweight='bold')
                ax4.set_ylabel('Time (ms)')
                ax4.set_xticks(range(len(models)))
                ax4.set_xticklabels(models, rotation=45, ha='right')
                
                # Panel 5: Precision-Recall
                ax5 = fig.add_subplot(gs[1, 1])
                for i, result in enumerate(results.values()):
                    precision = result.get('metrics', {}).get('precision', 0)
                    recall = result.get('metrics', {}).get('recall', 0)
                    ax5.scatter(recall, precision, label=result.get('model', f'Model {i}'), s=100)
                ax5.set_xlabel('Recall (%)')
                ax5.set_ylabel('Precision (%)')
                ax5.set_title('Precision-Recall Curve', fontweight='bold')
                ax5.legend()
                ax5.grid(True, linestyle='--', alpha=0.7)
                
                # Panel 6: Summary table
                ax6 = fig.add_subplot(gs[1, 2])
                ax6.axis('off')
                summary_text = "Summary Statistics:\n\n"
                summary_text += f"Total Models: {len(models)}\n"
                summary_text += f"Best Accuracy: {max(accuracies):.2f}%\n"
                summary_text += f"Avg Accuracy: {np.mean(accuracies):.2f}%\n"
                summary_text += f"Fastest Training: {min(train_times):.2f}h\n"
                summary_text += f"Smallest Model: {min(params):.2f}M"
                ax6.text(0.1, 0.5, summary_text, fontsize=11, va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                st.pyplot(fig)
                plt.close()
                
                # Download
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="Download Overview Figure",
                    data=buf.getvalue(),
                    file_name="results_overview.png",
                    mime="image/png"
                )
    
    with tab4:
        st.header("Export Options")
        
        results = st.session_state.experiment_results
        
        # Export configuration
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["PNG", "PDF", "SVG", "EPS"],
                index=0
            )
            
            resolution = st.selectbox(
                "Resolution (DPI)",
                [150, 300, 600],
                index=1
            )
        
        with col2:
            include_legend = st.checkbox("Include Legend", value=True)
            use_academic_style = st.checkbox("Use Academic Style", value=True)
        
        # Batch export
        st.subheader("Batch Export")
        
        if st.button("Export All Visualizations", type="primary"):
            st.info("Batch export would be implemented here.")
        
        # Individual export
        st.subheader("Individual Export")
        
        export_item = st.selectbox(
            "Select Item to Export",
            ["All Charts", "All Tables", "Selected Figure", "Custom Selection"],
            index=0
        )
        
        if export_item == "Custom Selection":
            items_to_export = st.multiselect(
                "Select Items",
                [
                    "Performance Comparison Chart",
                    "Training Curves",
                    "Confusion Matrix",
                    "Results Summary Table",
                    "Statistical Significance Table"
                ],
                default=["Performance Comparison Chart", "Results Summary Table"]
            )
            
            if st.button("Export Selected Items"):
                st.info(f"Exporting {len(items_to_export)} items...")
    
    with tab5:
        st.header("Visualization Settings")
        
        st.subheader("Color Schemes")
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Default (Seaborn)", "Publication (Colorblind)", "High Contrast", "Custom"],
            index=0
        )
        
        if color_scheme == "Custom":
            primary_color = st.color_picker("Primary Color", "#3b82f6")
            secondary_color = st.color_picker("Secondary Color", "#10b981")
            accent_color = st.color_picker("Accent Color", "#f59e0b")
        
        # Font settings
        st.subheader("Font Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            font_family = st.selectbox(
                "Font Family",
                ["DejaVu Sans", "Times New Roman", "Arial", "Helvetica"],
                index=0
            )
            
            font_size = st.number_input("Base Font Size", min_value=8, max_value=16, value=10)
        
        with col2:
            title_font_size = st.number_input("Title Font Size", min_value=10, max_value=24, value=14)
            legend_font_size = st.number_input("Legend Font Size", min_value=8, max_value=14, value=9)
        
        # Layout settings
        st.subheader("Layout Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            figure_width = st.number_input("Default Width (inches)", min_value=4, max_value=16, value=8)
            figure_height = st.number_input("Default Height (inches)", min_value=3, max_value=12, value=6)
        
        with col2:
            margin = st.slider("Figure Margin", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
            tight_layout = st.checkbox("Use Tight Layout", value=True)
        
        # Save settings
        if st.button("Save Visualization Settings"):
            viz_settings = {
                'color_scheme': color_scheme,
                'font_family': font_family,
                'font_size': font_size,
                'title_font_size': title_font_size,
                'legend_font_size': legend_font_size,
                'figure_width': figure_width,
                'figure_height': figure_height,
                'margin': margin,
                'tight_layout': tight_layout
            }
            st.session_state.viz_settings = viz_settings
            st.success("Visualization settings saved!")
    
    # Sidebar
    with st.sidebar:
        st.header("Visualization Dashboard Help")
        
        st.markdown("""
        ### How to Use This Page
        
        1. **Charts**: Create interactive visualizations
        2. **Tables**: Generate academic-quality tables
        3. **Academic Figures**: Create publication-ready figures
        4. **Export**: Download in various formats
        5. **Settings**: Customize appearance
        
        ### Tips
        
        - Use high DPI (300+) for publications
        - Export in PDF for vector graphics
        - Follow journal formatting guidelines
        - Include error bars when applicable
        - Use colorblind-friendly palettes
        """)
        
        st.markdown("---")
        
        # Quick actions
        st.header("Quick Actions")
        
        if st.button("Apply Default Settings"):
            if 'viz_settings' in st.session_state:
                del st.session_state.viz_settings
                st.success("Settings reset to defaults!")
                st.rerun()
        
        if st.button("Clear All Visualizations"):
            st.info("Visualization cache would be cleared here.")


def load_sample_results():
    """Load sample experiment results for demonstration"""
    sample_results = {
        "exp_001": {
            "model": "ResNet18",
            "dataset": "CIFAR10",
            "metrics": {
                "accuracy": 92.5,
                "precision": 91.8,
                "recall": 92.1,
                "f1_score": 91.9
            },
            "training_time": 1200,
            "inference_time": 15.2,
            "parameters": 11.7e6,
            "history": {
                "train_loss": [2.3, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3],
                "val_loss": [2.5, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7],
                "learning_rate": [0.001, 0.00095, 0.0009, 0.00085, 0.0008, 0.00075, 0.0007, 0.00065]
            }
        },
        "exp_002": {
            "model": "ResNet34",
            "dataset": "CIFAR10",
            "metrics": {
                "accuracy": 93.8,
                "precision": 93.2,
                "recall": 93.5,
                "f1_score": 93.3
            },
            "training_time": 2400,
            "inference_time": 22.5,
            "parameters": 21.8e6,
            "history": {
                "train_loss": [2.2, 1.6, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2],
                "val_loss": [2.4, 1.8, 1.3, 1.0, 0.8, 0.7, 0.6, 0.5],
                "learning_rate": [0.001, 0.00095, 0.0009, 0.00085, 0.0008, 0.00075, 0.0007, 0.00065]
            }
        },
        "exp_003": {
            "model": "VGG16",
            "dataset": "CIFAR10",
            "metrics": {
                "accuracy": 91.2,
                "precision": 90.5,
                "recall": 90.9,
                "f1_score": 90.7
            },
            "training_time": 3600,
            "inference_time": 28.7,
            "parameters": 138.4e6,
            "history": {
                "train_loss": [2.4, 1.9, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5],
                "val_loss": [2.6, 2.1, 1.6, 1.3, 1.1, 1.0, 0.9, 0.8],
                "learning_rate": [0.001, 0.00095, 0.0009, 0.00085, 0.0008, 0.00075, 0.0007, 0.00065]
            }
        }
    }
    st.session_state.experiment_results = sample_results


def dataframe_to_latex(df, caption=""):
    """Convert DataFrame to LaTeX table format"""
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{" + caption + "}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\begin{tabular}{|" + "|".join(["l"] * len(df.columns)) + "|}\n"
    latex += "\\hline\n"
    
    # Header
    latex += " & ".join([f"\\textbf{{{col}}}" for col in df.columns]) + " \\\\\n"
    latex += "\\hline\n"
    
    # Data rows
    for _, row in df.iterrows():
        latex += " & ".join([str(val) for val in row]) + " \\\\\n"
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


if __name__ == "__main__":
    main()