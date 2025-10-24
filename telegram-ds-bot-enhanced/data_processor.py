# data_processor.py
import os
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
import dask.dataframe as dd
import gc
import psutil
from scipy import stats
from ydata_profiling import ProfileReport
from model_builder import try_build_model, TargetDetector

# Configure matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configuration for optimized processing
CHUNK_SIZE = 10000
MAX_MEMORY_PERCENT = 75  # Maximum memory usage threshold

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise les types de données pour réduire l'utilisation mémoire"""
    df = df.copy()
    
    # Optimiser les entiers
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype(np.int16)
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype(np.int32)
    
    # Optimiser les flottants
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
    
    # Convertir les colonnes catégorielles
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].nunique() / len(df) < 0.5:  # Si moins de 50% de valeurs uniques
            df[col] = df[col].astype('category')
    
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Version optimisée du nettoyage de base avec traitement parallèle"""
    def clean_chunk(chunk):
        chunk = chunk.copy()
        chunk = chunk.drop_duplicates()
        
        # Traitement optimisé des valeurs manquantes
        for col in chunk.columns:
            if chunk[col].dtype.kind in 'biufc':
                median = chunk[col].median()
                chunk[col] = chunk[col].fillna(median if not pd.isna(median) else 0)
            else:
                mode = chunk[col].mode()
                chunk[col] = chunk[col].fillna(mode[0] if len(mode) else 'missing')
        return chunk
    
    # Vérifier la mémoire disponible
    mem = psutil.virtual_memory()
    if mem.percent > MAX_MEMORY_PERCENT:
        print(f"⚠️ Mémoire haute ({mem.percent}%), utilisation du traitement par chunks")
        # Utiliser dask pour le traitement par chunks
        ddf = dd.from_pandas(df, npartitions=os.cpu_count())
        df_cleaned = ddf.map_partitions(clean_chunk).compute()
    else:
        # Traitement parallèle si assez de mémoire
        chunks = np.array_split(df, os.cpu_count())
        cleaned_chunks = Parallel(n_jobs=-1)(delayed(clean_chunk)(chunk) for chunk in chunks)
        df_cleaned = pd.concat(cleaned_chunks, ignore_index=True)
    
    # Optimisation finale des types
    df_cleaned = optimize_dtypes(df_cleaned)
    
    # Force garbage collection
    gc.collect()
    
    return df_cleaned

def process_dataset(local_path: str, tmpdir: str, original_filename: str, preferences: dict = None) -> dict:
    name_lower = original_filename.lower()
    if name_lower.endswith('.csv'):
        df = pd.read_csv(local_path)
    else:
        df = pd.read_excel(local_path)

    if preferences is None:
        preferences = {'include_ml': True, 'include_gpt': True, 'include_advanced_stats': True}

    outputs = []
    
    # Store basic info
    row_count = df.shape[0]
    col_count = df.shape[1]
    
    # Clean
    cleaned = basic_clean(df)
    cleaned_path = os.path.join(tmpdir, 'cleaned_dataset.csv')
    cleaned.to_csv(cleaned_path, index=False)
    outputs.append(cleaned_path)

    # Enhanced EDA Profile with optimizations for large datasets
    # Limit sample size for very large datasets to prevent timeouts
    sample_size = min(1000, len(cleaned)) if len(cleaned) > 1000 else len(cleaned)
    sample_df = cleaned.sample(n=sample_size, random_state=42) if len(cleaned) > 1000 else cleaned
    
    profile = ProfileReport(
        sample_df, 
        title='Enhanced Data Analysis Report', 
        explorative=True,
        minimal=True,  # Use minimal mode for faster processing
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False}  # Disable Kendall for speed
        },
        missing_diagrams={
            "matrix": True,
            "bar": True,
            "heatmap": False  # Disable heatmap for speed
        },
        samples=None,  # Disable sample display for speed
        duplicates=None  # Disable duplicate detection for speed
    )
    profile_html = os.path.join(tmpdir, 'eda_profile.html')
    
    # Run profile generation in a separate process to avoid signal() limitations
    # and to isolate heavy CPU/memory usage. We join with timeout and terminate
    # the process if it exceeds the allowed time.
    from multiprocessing import Process

    def _profile_worker(df_slice, out_path):
        try:
            prof = ProfileReport(
                df_slice,
                title='Enhanced Data Analysis Report',
                explorative=True,
                minimal=True,
                correlations={
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False}
                },
                missing_diagrams={
                    "matrix": True,
                    "bar": True,
                    "heatmap": False
                },
                samples=None,
                duplicates=None
            )
            prof.to_file(out_path)
            print("✅ EDA profile generated successfully")
        except Exception as e:
            print(f"❌ Error generating EDA profile in subprocess: {e}")

    try:
        p = Process(target=_profile_worker, args=(sample_df, profile_html))
        p.start()
        # Wait up to 300 seconds (5 minutes)
        p.join(300)
        if p.is_alive():
            # Timed out; terminate the worker
            p.terminate()
            p.join(5)
            print("⚠️ EDA profile generation timed out, skipping...")
        else:
            # Check exitcode for errors
            if p.exitcode == 0:
                outputs.append(profile_html)
            else:
                print(f"⚠️ EDA profile subprocess exited with code {p.exitcode}")
    except Exception as e:
        print(f"❌ Error while running EDA profile subprocess: {e}")

    # Advanced statistical analysis if enabled
    if preferences.get('include_advanced_stats', True):
        try:
            advanced_stats = generate_advanced_statistics(cleaned, tmpdir)
            outputs.extend(advanced_stats)
        except Exception as e:
            logger.warning(f"Advanced statistics failed: {e}")

    # Try model building (optional, may be disabled via env or preferences)
    if preferences.get('include_ml', True):
        try:
            model_info = try_build_model(cleaned, tmpdir)
            # If AutoML is disabled, try_build_model returns a status dict like {'status': 'automl_disabled'}
            # Avoid creating a model_metrics file when AutoML is intentionally disabled.
            if model_info and not (isinstance(model_info, dict) and model_info.get('status') == 'automl_disabled'):
                metrics_path = os.path.join(tmpdir, 'model_metrics.txt')
                with open(metrics_path, 'w') as f:
                    f.write(str(model_info))
                outputs.append(metrics_path)
                if isinstance(model_info, dict) and model_info.get('model_path'):
                    outputs.append(model_info['model_path'])
            else:
                # Log that AutoML was skipped (no model files will be produced)
                logger.info('AutoML disabled or no model produced; skipping model artifacts')
        except Exception as e:
            err_path = os.path.join(tmpdir, 'model_error.txt')
            with open(err_path, 'w') as f:
                f.write(str(e))
            outputs.append(err_path)

    return {
        'files': outputs, 
        'summary_text': summarize_for_gpt(cleaned),
        'row_count': row_count,
        'col_count': col_count
    }

def generate_advanced_statistics(df: pd.DataFrame, tmpdir: str) -> list:
    """Generate advanced statistical analysis and visualizations"""
    outputs = []
    
    # Statistical tests summary
    stats_summary = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Normality tests
        stats_summary.append("=== NORMALITY TESTS ===")
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            try:
                shapiro_stat, shapiro_p = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
                stats_summary.append(f"{col}: Shapiro-Wilk p-value = {shapiro_p:.6f}")
            except:
                stats_summary.append(f"{col}: Normality test failed")
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            stats_summary.append("\n=== CORRELATION ANALYSIS ===")
            corr_matrix = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
            
            if high_corr:
                stats_summary.extend(high_corr[:10])  # Limit to top 10
            else:
                stats_summary.append("No high correlations found (|r| > 0.7)")
        
        # Outlier detection
        stats_summary.append("\n=== OUTLIER DETECTION ===")
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                stats_summary.append(f"{col}: {len(outliers)} outliers detected")
            except:
                stats_summary.append(f"{col}: Outlier detection failed")
    
    # Save statistics summary
    stats_path = os.path.join(tmpdir, 'advanced_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats_summary))
    outputs.append(stats_path)
    
    # Generate enhanced visualizations
    try:
        viz_files = create_enhanced_visualizations(df, tmpdir)
        outputs.extend(viz_files)
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    return outputs

def create_enhanced_visualizations(df: pd.DataFrame, tmpdir: str) -> list:
    """Create enhanced interactive visualizations with Plotly"""
    outputs = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # 1. Distribution plots for numeric columns
    if len(numeric_cols) > 0:
        # Distribution avec violin plot
        for col in numeric_cols[:3]:  # Limiter aux 3 premières colonnes
            fig = px.histogram(
                df, x=col,
                marginal="violin",
                title=f'Distribution de {col}',
                template="plotly_white",
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(
                showlegend=False,
                hovermode='x unified'
            )
            viz_path = os.path.join(tmpdir, f'distribution_{col}.html')
            fig.write_html(viz_path)
            outputs.append(viz_path)
        
        # Correlation heatmap interactive
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}'
            ))
            fig.update_layout(
                title='Matrice de Corrélation',
                template='plotly_white',
                height=800,
                width=800
            )
            corr_path = os.path.join(tmpdir, 'correlation_matrix.html')
            fig.write_html(corr_path)
            outputs.append(corr_path)
            
        # Scatter matrix for first 4 numeric columns
        if len(numeric_cols) >= 2:
            fig = px.scatter_matrix(
                df,
                dimensions=numeric_cols[:4],
                title='Matrice de Dispersion',
                template='plotly_white'
            )
            scatter_path = os.path.join(tmpdir, 'scatter_matrix.html')
            fig.write_html(scatter_path)
            outputs.append(scatter_path)
            
    # 2. Categorical analysis
    if len(categorical_cols) > 0:
        for col in categorical_cols[:3]:
            value_counts = df[col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution de {col}',
                template='plotly_white'
            )
            fig.update_layout(
                xaxis_title=col,
                yaxis_title='Fréquence',
                showlegend=False
            )
            cat_path = os.path.join(tmpdir, f'categorical_{col}.html')
            fig.write_html(cat_path)
            outputs.append(cat_path)
            
    # 3. Missing data visualization
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_pct = (missing_data / len(df) * 100).round(2)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=missing_pct.index,
            x=missing_pct.values,
            orientation='h',
            marker_color='#FF4B4B'
        ))
        fig.update_layout(
            title='Données Manquantes par Colonne',
            xaxis_title='Pourcentage Manquant (%)',
            template='plotly_white',
            height=max(400, len(df.columns) * 20)
        )
        missing_path = os.path.join(tmpdir, 'missing_data.html')
        fig.write_html(missing_path)
        outputs.append(missing_path)
    
    # 2. Categorical analysis
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(1, min(2, len(categorical_cols)), figsize=(15, 6))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        fig.suptitle('Categorical Data Analysis', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(categorical_cols[:2]):
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Top 10 Values in {col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        cat_plot_path = os.path.join(tmpdir, 'categorical_analysis.png')
        plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append(cat_plot_path)
    
    return outputs

def summarize_for_gpt(df: pd.DataFrame) -> str:
    # Enhanced summary for GPT input
    parts = []
    parts.append(f'Dataset Overview: {df.shape[0]} rows, {df.shape[1]} columns')
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    parts.append(f'Data Types: {dtype_counts.to_string()}')
    
    # Missing data summary
    nulls = df.isnull().sum().sort_values(ascending=False)
    nulls_pct = (nulls / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({'Missing Count': nulls, 'Missing %': nulls_pct})
    parts.append('Missing Data Summary:\n' + missing_summary[missing_summary['Missing Count'] > 0].to_string())
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].describe()
        parts.append('Numeric Summary:\n' + numeric_summary.to_string())
        
        # Correlation insights
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
            if high_corr:
                parts.append('High Correlations (|r| > 0.7):\n' + '\n'.join(high_corr[:5]))
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        parts.append('Categorical Summary:')
        for col in categorical_cols[:3]:  # Limit to first 3
            unique_count = df[col].nunique()
            top_value = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A'
            parts.append(f'  {col}: {unique_count} unique values, most common: {top_value}')
    
    return '\n\n'.join(parts)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
