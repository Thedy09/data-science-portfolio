"""
Enhanced AutoML Model Builder with improved architecture and best practices
"""
import os
import pandas as pd
import numpy as np
from joblib import dump
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Configure matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report, 
    confusion_matrix, r2_score, mean_absolute_error, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import json

# Configuration
ENABLE_AUTOML = os.getenv('ENABLE_AUTOML', 'false').lower() == 'true'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Structure pour stocker les résultats d'un modèle"""
    name: str
    model: Any
    metrics: Dict[str, float]
    cv_mean: float
    cv_std: float
    is_regression: bool


class DataPreprocessor:
    """Classe pour le prétraitement des données"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prétraite les features et la cible"""
        X_processed = self._preprocess_features(X, fit=True)
        y_processed = self._preprocess_target(y, fit=True)
        return X_processed, y_processed
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforme les données avec les encoders déjà fittés"""
        X_processed = self._preprocess_features(X, fit=False)
        y_processed = self._preprocess_target(y, fit=False) if y is not None else None
        return X_processed, y_processed
    
    def _preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Prétraite les features"""
        X_processed = X.copy()
        
        # Encode categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen labels
                    le = self.label_encoders[col]
                    X_processed[col] = X_processed[col].map(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        # Handle missing values intelligently
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        
        # Remove columns with zero variance
        if fit:
            variance = X_processed.var()
            self.feature_names = variance[variance > 0].index.tolist()
        
        X_processed = X_processed[self.feature_names]
        
        return X_processed.values
    
    def _preprocess_target(self, y: Optional[pd.Series], fit: bool = True) -> Optional[np.ndarray]:
        """Prétraite la variable cible"""
        if y is None:
            return None
            
        y_processed = y.copy()
        
        if y_processed.dtype == 'object' or y_processed.dtype.name == 'category':
            if fit:
                self.target_encoder = LabelEncoder()
                y_processed = self.target_encoder.fit_transform(y_processed.astype(str))
            else:
                y_processed = self.target_encoder.transform(y_processed.astype(str))
        
        # Handle missing values in target
        if pd.isna(y_processed).any():
            y_processed = pd.Series(y_processed).fillna(pd.Series(y_processed).median()).values
        
        return y_processed


class TargetDetector:
    """Classe pour détecter la meilleure colonne cible"""
    
    TARGET_KEYWORDS = ['target', 'label', 'y', 'class', 'outcome', 'result', 'prediction', 'output']
    
    @staticmethod
    def find_best_target(df: pd.DataFrame) -> Optional[str]:
        """Détecte la meilleure colonne cible avec des heuristiques multiples"""
        if df.empty:
            return None
        
        # Priority 1: Explicit target columns
        target_col = TargetDetector._find_by_keywords(df)
        if target_col:
            return target_col
        
        # Priority 2: Last column (common convention)
        last_col = df.columns[-1]
        if TargetDetector._is_valid_target(df[last_col]):
            return last_col
        
        # Priority 3: Binary columns (classification)
        binary_col = TargetDetector._find_binary_column(df)
        if binary_col:
            return binary_col
        
        # Priority 4: Categorical with reasonable classes
        categorical_col = TargetDetector._find_categorical_column(df)
        if categorical_col:
            return categorical_col
        
        # Priority 5: Numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[-1]  # Last numeric column
        
        return None
    
    @staticmethod
    def _find_by_keywords(df: pd.DataFrame) -> Optional[str]:
        """Trouve une colonne avec des mots-clés explicites"""
        lc = {c.lower(): c for c in df.columns}
        for keyword in TargetDetector.TARGET_KEYWORDS:
            if keyword in lc:
                return lc[keyword]
        return None
    
    @staticmethod
    def _find_binary_column(df: pd.DataFrame) -> Optional[str]:
        """Trouve une colonne binaire"""
        for col in df.columns:
            if df[col].nunique() == 2 and df[col].notna().sum() / len(df) > 0.9:
                return col
        return None
    
    @staticmethod
    def _find_categorical_column(df: pd.DataFrame) -> Optional[str]:
        """Trouve une colonne catégorielle avec un nombre raisonnable de classes"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:
                return col
        
        # Check numeric columns that might be categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20 and unique_count / len(df) < 0.05:
                return col
        
        return None
    
    @staticmethod
    def _is_valid_target(series: pd.Series) -> bool:
        """Vérifie si une série est une cible valide"""
        non_null_ratio = series.notna().sum() / len(series)
        return non_null_ratio > 0.7 and series.nunique() > 1


class ProblemTypeClassifier:
    """Classe pour déterminer le type de problème"""
    
    @staticmethod
    def is_regression(y: np.ndarray, threshold: float = 0.05) -> bool:
        """Détermine si le problème est une régression ou classification"""
        unique_values = len(np.unique(y))
        total_values = len(y)
        unique_ratio = unique_values / total_values
        
        # Heuristiques multiples
        if unique_values <= 2:
            return False  # Binary classification
        elif unique_values > 50:
            return True   # Likely regression
        elif unique_ratio < threshold:
            return False  # Classification
        else:
            return True   # Regression


class ModelTrainer:
    """Classe pour entraîner et comparer plusieurs modèles"""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
    
    def train_all_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        is_regression: bool
    ) -> List[ModelResult]:
        """Entraîne tous les modèles et retourne les résultats"""
        
        # Preprocess data
        X_processed, y_processed = self.preprocessor.fit_transform(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=TEST_SIZE, 
            random_state=self.random_state,
            stratify=y_processed if not is_regression and len(np.unique(y_processed)) > 1 else None
        )
        
        # Get model configurations
        model_configs = self._get_model_configs(is_regression)
        
        results = []
        for name, model, use_scaling in model_configs:
            try:
                result = self._train_single_model(
                    name, model, use_scaling,
                    X_train, X_test, y_train, y_test,
                    is_regression
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(f"⚠️  Failed to train {name}: {e}")
        
        return results
    
    def _get_model_configs(self, is_regression: bool) -> List[Tuple[str, Any, bool]]:
        """Retourne les configurations de modèles"""
        if is_regression:
            return [
                ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1), False),
                ('Gradient Boosting', GradientBoostingRegressor(random_state=self.random_state), False),
                ('Ridge Regression', Ridge(random_state=self.random_state), True),
                ('Lasso Regression', Lasso(random_state=self.random_state), True),
                ('SVR', SVR(), True)
            ]
        else:
            return [
                ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1), False),
                ('Gradient Boosting', GradientBoostingClassifier(random_state=self.random_state), False),
                ('Logistic Regression', LogisticRegression(random_state=self.random_state, max_iter=1000), True),
                ('SVC', SVC(random_state=self.random_state, probability=True), True)
            ]
    
    def _train_single_model(
        self, 
        name: str, 
        model: Any, 
        use_scaling: bool,
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray,
        is_regression: bool
    ) -> Optional[ModelResult]:
        """Entraîne un seul modèle"""
        
        # Scale if needed
        if use_scaling:
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used = X_train
            X_test_used = X_test
        
        # Train model
        model.fit(X_train_used, y_train)
        y_pred = model.predict(X_test_used)
        
        # Calculate metrics
        if is_regression:
            metrics = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
            cv_scoring = 'neg_mean_squared_error'
        else:
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
            }
            cv_scoring = 'accuracy'
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(
                model, X_train_used, y_train, 
                cv=min(CV_FOLDS, len(X_train_used)), 
                scoring=cv_scoring,
                n_jobs=-1
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
            
            # For regression, convert negative MSE back to positive
            if is_regression and cv_scoring == 'neg_mean_squared_error':
                cv_mean = -cv_mean
        except Exception as e:
            print(f"⚠️  Cross-validation failed for {name}: {e}")
            cv_mean = 0.0
            cv_std = 0.0
        
        return ModelResult(
            name=name,
            model=model,
            metrics=metrics,
            cv_mean=cv_mean,
            cv_std=cv_std,
            is_regression=is_regression
        )


class ModelSelector:
    """Classe pour sélectionner le meilleur modèle"""
    
    @staticmethod
    def select_best(results: List[ModelResult], out_dir: str) -> Dict[str, Any]:
        """Sélectionne le meilleur modèle"""
        if not results:
            return {'status': 'no_models_trained'}
        
        is_regression = results[0].is_regression
        
        # Select best model
        if is_regression:
            best_result = min(results, key=lambda x: x.metrics['mse'])
            primary_metric = 'mse'
            primary_value = best_result.metrics['mse']
        else:
            best_result = max(results, key=lambda x: x.metrics['accuracy'])
            primary_metric = 'accuracy'
            primary_value = best_result.metrics['accuracy']
        
        # Save best model
        model_path = os.path.join(out_dir, 'best_model.joblib')
        dump(best_result.model, model_path)
        
        # Prepare results
        result_dict = {
            'status': 'regression' if is_regression else 'classification',
            'best_model': best_result.name,
            primary_metric: primary_value,
            'all_metrics': best_result.metrics,
            'cv_mean': best_result.cv_mean,
            'cv_std': best_result.cv_std,
            'model_path': model_path,
            'all_models': [
                {
                    'name': r.name,
                    'metrics': r.metrics,
                    'cv_mean': r.cv_mean,
                    'cv_std': r.cv_std
                }
                for r in results
            ]
        }
        
        return result_dict


class Visualizer:
    """Classe pour créer des visualisations"""
    
    @staticmethod
    def create_visualizations(results: List[ModelResult], out_dir: str) -> List[str]:
        """Crée des visualisations de comparaison de modèles"""
        if not results:
            return []
        
        outputs = []
        is_regression = results[0].is_regression
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Model comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        model_names = [r.name for r in results]
        
        if is_regression:
            # Regression metrics
            mse_scores = [r.metrics['mse'] for r in results]
            r2_scores = [r.metrics['r2'] for r in results]
            
            axes[0].barh(model_names, mse_scores, alpha=0.7, color='coral')
            axes[0].set_xlabel('Mean Squared Error (Lower is Better)', fontsize=12)
            axes[0].set_title('Model Comparison - MSE', fontsize=14, fontweight='bold')
            axes[0].invert_xaxis()  # Lower is better
            
            axes[1].barh(model_names, r2_scores, alpha=0.7, color='green')
            axes[1].set_xlabel('R² Score (Higher is Better)', fontsize=12)
            axes[1].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
        else:
            # Classification metrics
            acc_scores = [r.metrics['accuracy'] for r in results]
            cv_means = [r.cv_mean for r in results]
            
            axes[0].barh(model_names, acc_scores, alpha=0.7, color='skyblue')
            axes[0].set_xlabel('Accuracy (Higher is Better)', fontsize=12)
            axes[0].set_title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
            axes[0].set_xlim([0, 1])
            
            axes[1].barh(model_names, cv_means, alpha=0.7, color='orange')
            axes[1].set_xlabel('CV Mean Score', fontsize=12)
            axes[1].set_title('Model Comparison - Cross-Validation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        comparison_path = os.path.join(out_dir, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append(comparison_path)
        
        return outputs


def try_build_model(df: pd.DataFrame, out_dir: str) -> Dict[str, Any]:
    """
    Point d'entrée principal pour construire un modèle automatiquement
    
    Args:
        df: DataFrame contenant les données
        out_dir: Répertoire de sortie pour sauvegarder les résultats
    
    Returns:
        Dictionnaire contenant les informations sur le modèle construit
    """
    if not ENABLE_AUTOML:
        return {'status': 'automl_disabled'}

    try:
        # Import model_explanation here to avoid circular imports
        from model_explanation import generate_model_insights
        
        # Validate input
        if df.empty or len(df.columns) < 2:
            return {'status': 'insufficient_data'}
        
        # Find target
        target = TargetDetector.find_best_target(df)
        if not target:
            return {'status': 'no_suitable_target_found'}
        
        print(f"✓ Target detected: {target}")
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target].copy()

        # Determine problem type
        y_temp = y.copy()
        if y_temp.dtype == 'object':
            y_temp = LabelEncoder().fit_transform(y_temp.astype(str))
        
        is_regression = ProblemTypeClassifier.is_regression(y_temp)
        print(f"✓ Problem type: {'Regression' if is_regression else 'Classification'}")
        
        # Train models
        trainer = ModelTrainer()
        results = trainer.train_all_models(X, y, is_regression)
        
        if not results:
            return {'status': 'no_models_trained'}
        
        print(f"✓ Trained {len(results)} models successfully")
        
        # Select best model
        best_model_info = ModelSelector.select_best(results, out_dir)
        print(f"✓ Best model: {best_model_info['best_model']}")
        
        # Create visualizations
        try:
            viz_files = Visualizer.create_visualizations(results, out_dir)
            best_model_info['visualization_files'] = viz_files
            print(f"✓ Created {len(viz_files)} visualization(s)")
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")
        
        # Generate detailed insights and explanations
        try:
            insights = generate_model_insights(best_model_info, out_dir)
            best_model_info['insights'] = insights
            print("✓ Generated detailed model insights and explanations")
            
            # Save insights to a file
            insights_path = os.path.join(out_dir, 'model_insights.txt')
            with open(insights_path, 'w', encoding='utf-8') as f:
                f.write("=== ANALYSE DÉTAILLÉE DES MODÈLES ===\n\n")
                f.write(f"Type de problème: {insights['problem_type']}\n")
                f.write(f"Meilleur modèle: {insights['best_model']}\n\n")
                
                f.write("EXPLICATION:\n")
                for exp in insights['explanation']:
                    f.write(f"- {exp}\n")
                
                f.write("\nCOMPARAISON DES MODÈLES:\n")
                for comp in insights['comparisons']:
                    f.write(f"\n{comp}\n")
                
                f.write("\nFORCES DU MODÈLE CHOISI:\n")
                for strength in insights['strengths']:
                    f.write(f"+ {strength}\n")
                
                f.write("\nPOINTS D'ATTENTION:\n")
                for weakness in insights['weaknesses']:
                    f.write(f"- {weakness}\n")
                
                f.write("\nRECOMMANDATIONS:\n")
                for rec in insights['recommendations']:
                    f.write(f"* {rec}\n")
            
            best_model_info['insights_file'] = insights_path
            
        except Exception as e:
            print(f"⚠️  Generation of insights failed: {e}")
        
        return best_model_info
    
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        return {'status': 'error', 'message': str(e)}


if __name__ == "__main__":
    # Example usage
    print("AutoML Model Builder - Enhanced Version")
    print("Set ENABLE_AUTOML=true environment variable to enable")