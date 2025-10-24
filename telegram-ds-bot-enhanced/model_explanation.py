"""
Module pour générer des explications détaillées des choix de modèles
"""
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import os

def generate_model_insights(model_results: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Génère des insights détaillés sur les performances des modèles"""
    
    is_regression = model_results['status'] == 'regression'
    best_model_name = model_results['best_model']
    all_models = model_results['all_models']
    
    insights = {
        'problem_type': 'regression' if is_regression else 'classification',
        'best_model': best_model_name,
        'explanation': [],
        'comparisons': [],
        'recommendations': []
    }
    
    # 1. Explication générale
    if is_regression:
        primary_metric = 'mse'
        insights['explanation'].append(
            f"Le problème a été identifié comme un problème de régression. "
            f"Le modèle {best_model_name} a été sélectionné comme le meilleur modèle avec "
            f"une erreur quadratique moyenne (MSE) de {model_results[primary_metric]:.4f}."
        )
    else:
        primary_metric = 'accuracy'
        insights['explanation'].append(
            f"Le problème a été identifié comme un problème de classification. "
            f"Le modèle {best_model_name} a été sélectionné comme le meilleur modèle avec "
            f"une précision de {model_results[primary_metric]*100:.2f}%."
        )
    
    # 2. Comparaison des modèles
    for model in all_models:
        comparison = analyze_model_performance(model, is_regression)
        insights['comparisons'].append(comparison)
    
    # 3. Forces et faiblesses du meilleur modèle
    best_model_info = next(m for m in all_models if m['name'] == best_model_name)
    strengths, weaknesses = analyze_model_characteristics(best_model_info, is_regression)
    insights['strengths'] = strengths
    insights['weaknesses'] = weaknesses
    
    # 4. Recommandations
    insights['recommendations'].extend(generate_recommendations(
        best_model_info,
        all_models,
        is_regression
    ))
    
    # 5. Visualisations supplémentaires
    try:
        viz_paths = create_detailed_visualizations(all_models, is_regression, out_dir)
        insights['visualizations'] = viz_paths
    except Exception as e:
        insights['visualizations'] = []
        print(f"Erreur lors de la création des visualisations : {e}")
    
    return insights

def analyze_model_performance(model: Dict[str, Any], is_regression: bool) -> str:
    """Analyse détaillée des performances d'un modèle"""
    if is_regression:
        return (
            f"Le modèle {model['name']} a obtenu:\n"
            f"- MSE: {model['metrics'].get('mse', 'N/A'):.4f}\n"
            f"- RMSE: {model['metrics'].get('rmse', 'N/A'):.4f}\n"
            f"- MAE: {model['metrics'].get('mae', 'N/A'):.4f}\n"
            f"- R²: {model['metrics'].get('r2', 'N/A'):.4f}\n"
            f"- Validation croisée: {model['cv_mean']:.4f} (±{model['cv_std']:.4f})"
        )
    else:
        return (
            f"Le modèle {model['name']} a obtenu:\n"
            f"- Précision: {model['metrics'].get('accuracy', 'N/A')*100:.2f}%\n"
            f"- F1-score: {model['metrics'].get('f1_score', 'N/A'):.4f}\n"
            f"- Validation croisée: {model['cv_mean']*100:.2f}% (±{model['cv_std']*100:.2f}%)"
        )

def analyze_model_characteristics(model: Dict[str, Any], is_regression: bool) -> tuple:
    """Analyse les forces et faiblesses du modèle"""
    strengths = []
    weaknesses = []
    
    # Analyse basée sur le type de modèle
    if 'Random Forest' in model['name']:
        strengths.extend([
            "Bonne gestion des relations non linéaires",
            "Robuste aux outliers et aux valeurs manquantes",
            "Capture automatiquement les interactions entre variables"
        ])
        weaknesses.extend([
            "Peut être plus lent à l'entraînement que des modèles plus simples",
            "Moins interprétable que des modèles linéaires"
        ])
    
    elif 'Gradient Boosting' in model['name']:
        strengths.extend([
            "Performances généralement excellentes",
            "Bonne gestion des relations complexes",
            "Apprentissage séquentiel qui réduit le biais"
        ])
        weaknesses.extend([
            "Risque de surapprentissage si mal paramétré",
            "Temps d'entraînement plus long",
            "Sensible au bruit dans les données"
        ])
    
    elif 'Linear' in model['name'] or 'Logistic' in model['name']:
        strengths.extend([
            "Très interprétable",
            "Rapide à entraîner",
            "Bonnes performances sur des relations linéaires"
        ])
        weaknesses.extend([
            "Ne capture pas les relations non linéaires",
            "Sensible aux outliers",
            "Hypothèses fortes sur la distribution des données"
        ])
    
    elif 'SV' in model['name']:  # SVC ou SVR
        strengths.extend([
            "Efficace dans les espaces de grande dimension",
            "Bonne gestion des relations non linéaires via les kernels",
            "Robuste au surapprentissage"
        ])
        weaknesses.extend([
            "Temps de calcul important sur de grands datasets",
            "Sensible au choix des hyperparamètres",
            "Moins performant si beaucoup de bruit dans les données"
        ])
    
    # Analyse basée sur les métriques
    cv_ratio = model['cv_std'] / (model['cv_mean'] if model['cv_mean'] != 0 else 1)
    if cv_ratio < 0.1:
        strengths.append("Très stable entre les différents folds de validation croisée")
    elif cv_ratio > 0.3:
        weaknesses.append("Performance variable entre les folds de validation croisée")
    
    if is_regression:
        r2 = model['metrics'].get('r2', 0)
        if r2 > 0.9:
            strengths.append("Excellent ajustement aux données (R² > 0.9)")
        elif r2 < 0.5:
            weaknesses.append("Ajustement limité aux données (R² < 0.5)")
    else:
        acc = model['metrics'].get('accuracy', 0)
        if acc > 0.9:
            strengths.append("Très haute précision (> 90%)")
        elif acc < 0.6:
            weaknesses.append("Précision limitée (< 60%)")
    
    return strengths, weaknesses

def generate_recommendations(best_model: Dict[str, Any], all_models: List[Dict[str, Any]], is_regression: bool) -> List[str]:
    """Génère des recommandations pour améliorer les performances"""
    recommendations = []
    
    # Analyse de la stabilité
    cv_ratio = best_model['cv_std'] / (best_model['cv_mean'] if best_model['cv_mean'] != 0 else 1)
    if cv_ratio > 0.2:
        recommendations.append(
            "La variabilité des performances en validation croisée est importante. "
            "Considérez augmenter la taille du dataset ou utiliser des techniques de régularisation."
        )
    
    # Analyse des performances relatives
    performances = []
    for model in all_models:
        if is_regression:
            perf = -model['metrics'].get('mse', float('inf'))  # Négatif car plus petit est meilleur
        else:
            perf = model['metrics'].get('accuracy', 0)
        performances.append((model['name'], perf))
    
    performances.sort(key=lambda x: x[1], reverse=True)
    
    # Si le meilleur modèle n'est pas largement supérieur aux autres
    if len(performances) > 1:
        best_perf = performances[0][1]
        second_perf = performances[1][1]
        if abs(best_perf - second_perf) / abs(best_perf) < 0.1:  # Différence < 10%
            recommendations.append(
                f"Les performances du {performances[0][0]} et du {performances[1][0]} sont proches. "
                f"Vous pourriez essayer un ensemble de ces deux modèles."
            )
    
    # Recommandations spécifiques au type de modèle
    if 'Random Forest' in best_model['name'] or 'Gradient Boosting' in best_model['name']:
        recommendations.append(
            "Pour améliorer les performances, vous pourriez essayer:\n"
            "- Augmenter le nombre d'arbres (n_estimators)\n"
            "- Ajuster la profondeur maximale (max_depth)\n"
            "- Faire de la sélection de features"
        )
    elif 'SV' in best_model['name']:
        recommendations.append(
            "Pour améliorer les performances, vous pourriez essayer:\n"
            "- Différents kernels (linear, rbf, polynomial)\n"
            "- Ajuster le paramètre de régularisation C\n"
            "- Normaliser les features"
        )
    
    # Recommandations générales
    if is_regression:
        if best_model['metrics'].get('r2', 1) < 0.5:
            recommendations.append(
                "Le R² est assez faible. Considérez:\n"
                "- Ajouter des features plus informatives\n"
                "- Transformer les variables (log, polynomial, etc.)\n"
                "- Utiliser des techniques de sélection de variables"
            )
    else:
        if best_model['metrics'].get('accuracy', 1) < 0.7:
            recommendations.append(
                "La précision pourrait être améliorée. Considérez:\n"
                "- Techniques de resampling pour les classes déséquilibrées\n"
                "- Feature engineering\n"
                "- Collecte de plus de données"
            )
    
    return recommendations

def create_detailed_visualizations(models: List[Dict[str, Any]], is_regression: bool, out_dir: str) -> List[str]:
    """Crée des visualisations détaillées des performances des modèles"""
    viz_paths = []
    
    # 1. Comparaison des métriques principales
    model_names = [m['name'] for m in models]
    if is_regression:
        metrics = {
            'MSE': [m['metrics'].get('mse', float('nan')) for m in models],
            'R²': [m['metrics'].get('r2', float('nan')) for m in models],
            'MAE': [m['metrics'].get('mae', float('nan')) for m in models]
        }
    else:
        metrics = {
            'Accuracy': [m['metrics'].get('accuracy', float('nan')) for m in models],
            'F1-Score': [m['metrics'].get('f1_score', float('nan')) for m in models]
        }
    
    # Créer un graphique pour chaque métrique
    for metric_name, values in metrics.items():
        fig = go.Figure(data=[
            go.Bar(name=metric_name, x=model_names, y=values,
                  text=[f'{v:.4f}' for v in values],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=f'Comparaison des modèles - {metric_name}',
            xaxis_title="Modèles",
            yaxis_title=metric_name,
            template='plotly_white'
        )
        
        path = os.path.join(out_dir, f'comparison_{metric_name.lower().replace(" ", "_")}.html')
        fig.write_html(path)
        viz_paths.append(path)
    
    # 2. Stabilité des modèles (validation croisée)
    cv_means = [m['cv_mean'] for m in models]
    cv_stds = [m['cv_std'] for m in models]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Performance moyenne',
        x=model_names,
        y=cv_means,
        error_y=dict(type='data', array=cv_stds, visible=True)
    ))
    
    fig.update_layout(
        title='Stabilité des modèles (Validation croisée)',
        xaxis_title="Modèles",
        yaxis_title="Score moyen",
        template='plotly_white'
    )
    
    path = os.path.join(out_dir, 'model_stability.html')
    fig.write_html(path)
    viz_paths.append(path)
    
    return viz_paths