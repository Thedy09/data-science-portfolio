# ... [Contenu précédent inchangé] ...

def try_html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """
    Fonction de compatibilité pour la conversion HTML vers PDF
    
    Args:
        html_path: Chemin vers le fichier HTML
        pdf_path: Chemin de sortie du PDF
    
    Returns:
        bool: True si succès, False sinon
    """
    converter = PDFConverter()
    success, error = converter.convert(html_path, pdf_path)
    if error:
        print(f"PDF conversion failed: {error}")
    return success


def bundle_outputs(file_paths: List[str], zip_path: str) -> None:
    """
    Crée un bundle des fichiers d'analyse
    
    Args:
        file_paths: Liste des chemins de fichiers à inclure
        zip_path: Chemin du fichier ZIP à créer
    """
    try:
        # Créer un bundle des fichiers
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Ajouter chaque fichier au ZIP
            for file_path in file_paths:
                if os.path.exists(file_path):
                    # Utiliser seulement le nom du fichier comme nom dans le ZIP
                    arcname = os.path.basename(file_path)
                    zf.write(file_path, arcname)

            # Créer et ajouter le README
            readme_content = _generate_readme(file_paths)
            zf.writestr('README.md', readme_content)

            # Créer et ajouter les métadonnées
            metadata = AnalysisMetadata.from_files(file_paths)
            metadata_content = json.dumps(metadata.to_dict(), indent=2)
            zf.writestr('metadata.json', metadata_content)

        print(f"✓ Bundle créé avec succès: {os.path.basename(zip_path)}")
        print(f"  Taille: {_format_size(os.path.getsize(zip_path))}")

    except Exception as e:
        raise Exception(f"Erreur lors de la création du bundle: {str(e)}")


def _generate_readme(file_paths: List[str]) -> str:
    """Génère le contenu du README pour le bundle"""
    template = """# 📊 Résultats d'Analyse de Données

Ce package contient les résultats complets de l'analyse automatique de données.

## 📁 Contenu

{file_list}

## 🚀 Pour Commencer

1. Ouvrez `eda_profile.html` dans votre navigateur pour le rapport interactif
2. Consultez les visualisations dans le dossier `figures/`
3. Les données nettoyées sont dans `cleaned_dataset.csv`
4. Le modèle est sauvegardé dans `best_model.joblib`

## 📈 Visualisations

Les visualisations incluent :
- Distribution des variables
- Corrélations
- Analyse des valeurs manquantes
- Comparaison des modèles

## 🤖 Modèle Machine Learning

Le modèle choisi a été optimisé et validé avec :
- Validation croisée
- Optimisation des hyperparamètres
- Tests de performance rigoureux

## 📝 Notes

- Date de génération : {date}
- Version : {version}
- Nombre de fichiers : {file_count}

---

Généré par Data Science Bot Enhanced"""

    # Créer la liste des fichiers
    file_list = []
    for path in file_paths:
        if os.path.exists(path):
            name = os.path.basename(path)
            size = _format_size(os.path.getsize(path))
            file_list.append(f"- `{name}` ({size})")

    # Remplir le template
    content = template.format(
        file_list="\n".join(file_list),
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version="2.1",
        file_count=len(file_paths)
    )

    return content


def _format_size(size: int) -> str:
    """Formate une taille en bytes de manière lisible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


if __name__ == "__main__":
    print("Report Generator - Enhanced Version v2.1")
    print("=" * 50)
    print("\nFeatures:")
    print("  ✓ PDF conversion with fallback")
    print("  ✓ Comprehensive summary reports")
    print("  ✓ Metadata generation")
    print("  ✓ File categorization")
    print("  ✓ Professional README generation")