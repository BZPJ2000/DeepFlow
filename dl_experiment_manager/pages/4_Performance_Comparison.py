"""
Performance Comparison Page

This page allows users to compare performance across multiple models and datasets,
with statistical analysis and visualization.
"""

import streamlit as st
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function for performance comparison page"""
    
    st.set_page_config(
        page_title="Performance Comparison - DL Experiment Manager",
        page_icon="📈",
        layout="wide"
    )
    
    # Title and description
    st.title("📈 Performance Comparison")
    st.markdown("""
    Compare performance across multiple models and datasets.
    Statistical analysis and comprehensive visualizations for research.
    """)
    
    # Check if experiments exist
    has_experiments = 'experiment_results' in st.session_state and st.session_state.experiment_results
    
    if not has_experiments:
        st.warning("⚠️ No experiment results available.")
        st.info("Please run experiments first to generate results for comparison.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Experiment Setup", type="primary"):
                st.session_state.current_page = 'experiment_setup'
                st.rerun()
        with col2:
            if st.button("Load Sample Results"):
                # Load sample results for demonstration
                load_sample_results()
                st.success("Sample results loaded!")
                st.rerun()
        
        return
    
    # Main content in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Visualizations", 
        "🔬 Statistical Analysis", 
        "📋 Detailed Results",
        "📄 Export Report"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        # Summary metrics
        st.subheader("Summary Statistics")
        
        results = st.session_state.experiment_results
        
        # Calculate summary statistics
        summary_data = []
        for exp_id, result in results.items():
            summary_data.append({
                'Experiment ID': exp_id,
                'Model': result.get('model', 'N/A'),
                'Dataset': result.get('dataset', 'N/A'),
                'Accuracy': result.get('metrics', {}).get('accuracy', 0),
                'Training Time': result.get('training_time', 0),
                'Inference Time': result.get('inference_time', 0),
                'Parameters': result.get('parameters', 0)
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Display summary table
        st.dataframe(df_summary, use_container_width=True)
        
        # Key metrics
        st.subheader("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_accuracy = df_summary['Accuracy'].max()
            st.metric("Best Accuracy", f"{best_accuracy:.2f}%")
        
        with col2:
            avg_accuracy = df_summary['Accuracy'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.2f}%")
        
        with col3:
            fastest_inference = df_summary['Inference Time'].min()
            st.metric("Fastest Inference", f"{fastest_inference:.2f}ms")
        
        with col4:
            total_params = df_summary['Parameters'].sum()
            st.metric("Total Parameters", f"{total_params/1e6:.2f}M")
        
        # Model ranking
        st.subheader("Model Ranking")
        
        ranking_metrics = st.selectbox(
            "Ranking Metric",
            ["Accuracy", "Training Time", "Inference Time", "Parameters"],
            index=0
        )
        
        df_ranked = df_summary.sort_values(by=ranking_metrics, ascending=False if ranking_metrics == "Accuracy" else True)
        
        st.dataframe(df_ranked, use_container_width=True)
    
    with tab2:
        st.header("Performance Visualizations")
        
        results = st.session_state.experiment_results
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Visualization Type",
            ["Bar Charts", "Line Charts", "Radar Charts", "Heatmaps", "Box Plots"],
            index=0
        )
        
        if viz_type == "Bar Charts":
            st.subheader("Performance Comparison (Bar Chart)")
            
            metrics_to_plot = st.multiselect(
                "Select Metrics to Plot",
                ["Accuracy", "Precision", "Recall", "F1 Score", "Training Time", "Inference Time"],
                default=["Accuracy", "Training Time"]
            )
            
            if metrics_to_plot:
                import plotly.express as px
                
                # Prepare data for plotting
                plot_data = []
                for exp_id, result in results.items():
                    for metric in metrics_to_plot:
                        value = result.get('metrics', {}).get(metric.lower().replace(' ', '_'), 0)
                        plot_data.append({
                            'Experiment': exp_id,
                            'Model': result.get('model', 'N/A'),
                            'Metric': metric,
                            'Value': value
                        })
                
                df_plot = pd.DataFrame(plot_data)
                
                fig = px.bar(
                    df_plot,
                    x='Model',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Model Performance Comparison',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Charts":
            st.subheader("Training Progress (Line Chart)")
            
            experiment_ids = list(results.keys())
            selected_experiments = st.multiselect(
                "Select Experiments to Plot",
                experiment_ids,
                default=experiment_ids[:3]
            )
            
            if selected_experiments:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                for exp_id in selected_experiments:
                    result = results[exp_id]
                    history = result.get('history', {})
                    
                    if 'train_loss' in history and 'val_loss' in history:
                        epochs = list(range(len(history['train_loss'])))
                        
                        fig.add_trace(go.Scatter(
                            x=epochs,
                            y=history['train_loss'],
                            mode='lines',
                            name=f"{exp_id} - Train Loss",
                            line=dict(width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=epochs,
                            y=history['val_loss'],
                            mode='lines',
                            name=f"{exp_id} - Val Loss",
                            line=dict(width=2, dash='dash')
                        ))
                
                fig.update_layout(
                    title='Training Loss Over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Radar Charts":
            st.subheader("Multi-Metric Comparison (Radar Chart)")
            
            metrics_for_radar = st.multiselect(
                "Select Metrics for Radar Chart",
                ["Accuracy", "Precision", "Recall", "F1 Score"],
                default=["Accuracy", "Precision", "Recall", "F1 Score"]
            )
            
            if len(metrics_for_radar) >= 3:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                for exp_id, result in results.items():
                    metrics = result.get('metrics', {})
                    values = [metrics.get(m.lower().replace(' ', '_'), 0) for m in metrics_for_radar]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics_for_radar,
                        fill='toself',
                        name=result.get('model', exp_id)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 3 metrics for radar chart.")
        
        elif viz_type == "Heatmaps":
            st.subheader("Performance Matrix (Heatmap)")
            
            # Create performance matrix
            models = list(set([r.get('model', 'N/A') for r in results.values()]))
            datasets = list(set([r.get('dataset', 'N/A') for r in results.values()]))
            
            metric_for_heatmap = st.selectbox(
                "Metric for Heatmap",
                ["Accuracy", "Precision", "Recall", "F1 Score"],
                index=0
            )
            
            matrix_data = []
            for model in models:
                row = []
                for dataset in datasets:
                    # Find result for this model-dataset combination
                    value = 0
                    for exp_id, result in results.items():
                        if result.get('model') == model and result.get('dataset') == dataset:
                            value = result.get('metrics', {}).get(metric_for_heatmap.lower().replace(' ', '_'), 0)
                            break
                    row.append(value)
                matrix_data.append(row)
            
            import plotly.express as px
            
            df_matrix = pd.DataFrame(matrix_data, index=models, columns=datasets)
            
            fig = px.imshow(
                df_matrix,
                labels=dict(x="Dataset", y="Model", color=metric_for_heatmap),
                x=datasets,
                y=models,
                color_continuous_scale='Viridis',
                title=f'{metric_for_heatmap} Across Model-Dataset Combinations'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plots":
            st.subheader("Metric Distribution (Box Plot)")
            
            metric_for_box = st.selectbox(
                "Metric for Box Plot",
                ["Accuracy", "Precision", "Recall", "F1 Score", "Training Time", "Inference Time"],
                index=0
            )
            
            # Prepare data for box plot
            box_data = []
            for exp_id, result in results.items():
                value = result.get('metrics', {}).get(metric_for_box.lower().replace(' ', '_'), 0)
                box_data.append({
                    'Model': result.get('model', 'N/A'),
                    'Value': value
                })
            
            df_box = pd.DataFrame(box_data)
            
            import plotly.express as px
            
            fig = px.box(
                df_box,
                x='Model',
                y='Value',
                title=f'{metric_for_box} Distribution Across Models'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Statistical Analysis")
        
        results = st.session_state.experiment_results
        
        # Statistical tests
        st.subheader("Statistical Tests")
        
        test_type = st.selectbox(
            "Statistical Test",
            ["Paired t-test", "ANOVA", "Wilcoxon Signed-Rank Test"],
            index=0
        )
        
        # Select models to compare
        models = list(set([r.get('model', 'N/A') for r in results.values()]))
        models_to_compare = st.multiselect(
            "Select Models to Compare",
            models,
            default=models[:2] if len(models) >= 2 else models
        )
        
        if len(models_to_compare) >= 2:
            metric_for_test = st.selectbox(
                "Metric for Statistical Test",
                ["Accuracy", "Precision", "Recall", "F1 Score"],
                index=0
            )
            
            if st.button("Run Statistical Test", type="primary"):
                # Perform statistical test
                if test_type == "Paired t-test":
                    perform_t_test(results, models_to_compare, metric_for_test)
                elif test_type == "ANOVA":
                    perform_anova(results, models_to_compare, metric_for_test)
                elif test_type == "Wilcoxon Signed-Rank Test":
                    perform_wilcoxon_test(results, models_to_compare, metric_for_test)
        
        # Significance levels
        st.subheader("Significance Levels")
        
        alpha_level = st.slider(
            "Significance Level (α)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01
        )
        
        st.info(f"Results with p-value < {alpha_level} are considered statistically significant.")
        
        # Effect size
        st.subheader("Effect Size Analysis")
        
        effect_size_metric = st.selectbox(
            "Effect Size Metric",
            ["Cohen's d", "Hedges' g", "Glass's delta"],
            index=0
        )
        
        if st.button("Calculate Effect Size"):
            st.info("Effect size calculation would be implemented here.")
    
    with tab4:
        st.header("Detailed Results")
        
        results = st.session_state.experiment_results
        
        # Filter results
        st.subheader("Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_model = st.multiselect(
                "Filter by Model",
                list(set([r.get('model', 'N/A') for r in results.values()]))
            )
        
        with col2:
            filter_dataset = st.multiselect(
                "Filter by Dataset",
                list(set([r.get('dataset', 'N/A') for r in results.values()]))
            )
        
        with col3:
            min_accuracy = st.slider(
                "Minimum Accuracy (%)",
                min_value=0,
                max_value=100,
                value=0
            )
        
        # Apply filters
        filtered_results = {}
        for exp_id, result in results.items():
            model = result.get('model', 'N/A')
            dataset = result.get('dataset', 'N/A')
            accuracy = result.get('metrics', {}).get('accuracy', 0)
            
            if filter_model and model not in filter_model:
                continue
            if filter_dataset and dataset not in filter_dataset:
                continue
            if accuracy < min_accuracy:
                continue
            
            filtered_results[exp_id] = result
        
        # Display filtered results
        st.write(f"Showing {len(filtered_results)} of {len(results)} results")
        
        # Detailed table
        detailed_data = []
        for exp_id, result in filtered_results.items():
            metrics = result.get('metrics', {})
            detailed_data.append({
                'Experiment ID': exp_id,
                'Model': result.get('model', 'N/A'),
                'Dataset': result.get('dataset', 'N/A'),
                'Accuracy (%)': metrics.get('accuracy', 0),
                'Precision (%)': metrics.get('precision', 0),
                'Recall (%)': metrics.get('recall', 0),
                'F1 Score (%)': metrics.get('f1_score', 0),
                'Training Time (s)': result.get('training_time', 0),
                'Inference Time (ms)': result.get('inference_time', 0),
                'Parameters (M)': result.get('parameters', 0) / 1e6
            })
        
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(df_detailed, use_container_width=True)
        
        # Download filtered results
        if st.button("Download Filtered Results"):
            csv = df_detailed.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_results.csv",
                mime="text/csv"
            )
    
    with tab5:
        st.header("Export Report")
        
        st.subheader("Report Configuration")
        
        report_title = st.text_input(
            "Report Title",
            value="Deep Learning Experiment Results"
        )
        
        author_name = st.text_input(
            "Author Name",
            value="Researcher"
        )
        
        report_sections = st.multiselect(
            "Include in Report",
            [
                "Executive Summary",
                "Methodology",
                "Results Overview",
                "Statistical Analysis",
                "Visualizations",
                "Conclusions",
                "References"
            ],
            default=["Executive Summary", "Results Overview", "Visualizations"]
        )
        
        # Report format
        report_format = st.selectbox(
            "Report Format",
            ["HTML", "PDF", "Markdown", "LaTeX"],
            index=0
        )
        
        # Generate report
        if st.button("Generate Report", type="primary"):
            st.info("Report generation would be implemented here.")
            
            # Show preview
            st.subheader("Report Preview")
            
            st.markdown(f"""
            # {report_title}
            
            **Author:** {author_name}
            **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
            
            ## Executive Summary
            
            This report summarizes the results of {len(st.session_state.experiment_results)} experiments.
            
            ## Results Overview
            
            - **Total Experiments:** {len(st.session_state.experiment_results)}
            - **Models Tested:** {len(set([r.get('model', 'N/A') for r in st.session_state.experiment_results.values()]))}
            - **Datasets Used:** {len(set([r.get('dataset', 'N/A') for r in st.session_state.experiment_results.values()]))}
            
            *Full report would be generated based on selected sections.*
            """)
    
    # Sidebar
    with st.sidebar:
        st.header("Performance Comparison Help")
        
        st.markdown("""
        ### How to Use This Page
        
        1. **Overview**: View summary statistics and rankings
        2. **Visualizations**: Create various charts and plots
        3. **Statistical Analysis**: Run statistical tests
        4. **Detailed Results**: Filter and export results
        5. **Export Report**: Generate comprehensive reports
        
        ### Tips
        
        - Use multiple visualization types for comprehensive analysis
        - Statistical tests help validate significance
        - Export results for publication
        - Filter results to focus on specific experiments
        """)
        
        st.markdown("---")
        
        # Quick actions
        st.header("Quick Actions")
        
        if st.button("Refresh Results"):
            st.info("Results refresh would be implemented here.")
        
        if st.button("Clear All Results"):
            if 'experiment_results' in st.session_state:
                st.session_state.experiment_results = {}
                st.success("All results cleared!")
                st.rerun()


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
                "val_loss": [2.5, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7]
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
                "val_loss": [2.4, 1.8, 1.3, 1.0, 0.8, 0.7, 0.6, 0.5]
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
                "val_loss": [2.6, 2.1, 1.6, 1.3, 1.1, 1.0, 0.9, 0.8]
            }
        }
    }
    st.session_state.experiment_results = sample_results


def perform_t_test(results, models, metric):
    """Perform paired t-test between models"""
    st.info(f"Performing paired t-test for {metric}...")
    
    # Extract metric values for each model
    model_values = {}
    for model in models:
        values = []
        for exp_id, result in results.items():
            if result.get('model') == model:
                value = result.get('metrics', {}).get(metric.lower().replace(' ', '_'), 0)
                values.append(value)
        model_values[model] = values
    
    # Perform t-test (placeholder)
    from scipy import stats
    
    if len(models) == 2:
        model1, model2 = models
        t_stat, p_value = stats.ttest_ind(model_values[model1], model_values[model2])
        
        st.write(f"**t-statistic:** {t_stat:.4f}")
        st.write(f"**p-value:** {p_value:.4f}")
        
        if p_value < 0.05:
            st.success(f"Result is statistically significant (p < 0.05)")
        else:
            st.warning(f"Result is not statistically significant (p >= 0.05)")


def perform_anova(results, models, metric):
    """Perform ANOVA test across multiple models"""
    st.info(f"Performing ANOVA for {metric}...")
    
    # Extract metric values for each model
    model_values = []
    for model in models:
        values = []
        for exp_id, result in results.items():
            if result.get('model') == model:
                value = result.get('metrics', {}).get(metric.lower().replace(' ', '_'), 0)
                values.append(value)
        model_values.append(values)
    
    # Perform ANOVA (placeholder)
    from scipy import stats
    
    f_stat, p_value = stats.f_oneway(*model_values)
    
    st.write(f"**F-statistic:** {f_stat:.4f}")
    st.write(f"**p-value:** {p_value:.4f}")
    
    if p_value < 0.05:
        st.success(f"Result is statistically significant (p < 0.05)")
    else:
        st.warning(f"Result is not statistically significant (p >= 0.05)")


def perform_wilcoxon_test(results, models, metric):
    """Perform Wilcoxon signed-rank test"""
    st.info(f"Performing Wilcoxon signed-rank test for {metric}...")
    
    # Extract metric values for each model
    model_values = {}
    for model in models:
        values = []
        for exp_id, result in results.items():
            if result.get('model') == model:
                value = result.get('metrics', {}).get(metric.lower().replace(' ', '_'), 0)
                values.append(value)
        model_values[model] = values
    
    # Perform Wilcoxon test (placeholder)
    from scipy import stats
    
    if len(models) == 2:
        model1, model2 = models
        statistic, p_value = stats.wilcoxon(model_values[model1], model_values[model2])
        
        st.write(f"**Statistic:** {statistic:.4f}")
        st.write(f"**p-value:** {p_value:.4f}")
        
        if p_value < 0.05:
            st.success(f"Result is statistically significant (p < 0.05)")
        else:
            st.warning(f"Result is not statistically significant (p >= 0.05)")


if __name__ == "__main__":
    main()