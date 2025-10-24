"""
Enhanced Report Generator with improved architecture and features
"""
import os
import zipfile
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Optional imports with graceful fallback
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("⚠️  pdfkit not available. PDF generation will be skipped.")


class FileCategory(Enum):
    """Catégories de fichiers pour l'organisation"""
    DATA = ('Data Files', ['.csv', '.xlsx', '.json', '.parquet'])
    REPORTS = ('Reports', ['.html', '.pdf', '.md'])
    VISUALIZATIONS = ('Visualizations', ['.png', '.jpg', '.jpeg', '.svg'])
    MODELS = ('Model Files', ['.joblib', '.pkl', '.h5', '.pt'])
    STATISTICS = ('Statistics', ['.txt', '.json'])
    AI_INSIGHTS = ('AI Insights', ['gpt_summary.txt', 'ai_summary.txt', 'insights.txt'])
    
    def __init__(self, display_name: str, extensions: List[str]):
        self.display_name = display_name
        self.extensions = extensions
    
    @classmethod
    def categorize_file(cls, filepath: str) -> Optional['FileCategory']:
        """Catégorise un fichier selon son extension"""
        filename = os.path.basename(filepath).lower()
        
        for category in cls:
            for ext in category.extensions:
                if filename.endswith(ext.lower()):
                    return category
        return None


@dataclass
class AnalysisMetadata:
    """Métadonnées complètes de l'analyse"""
    generated_at: str
    total_files: int
    file_types: Dict[str, int]
    file_sizes: Dict[str, int]
    analysis_version: str
    zip_size: Optional[int] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire"""
        return asdict(self)
    
    @classmethod
    def from_files(cls, file_paths: List[str], version: str = '2.1') -> 'AnalysisMetadata':
        """Crée les métadonnées à partir d'une liste de fichiers"""
        file_types = {}
        file_sizes = {}
        
        for filepath in file_paths:
            if os.path.exists(filepath):
                # Type de fichier
                ext = os.path.splitext(filepath)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                
                # Taille de fichier
                size = os.path.getsize(filepath)
                file_sizes[os.path.basename(filepath)] = size
        
        return cls(
            generated_at=datetime.now().isoformat(),
            total_files=len(file_paths),
            file_types=file_types,
            file_sizes=file_sizes,
            analysis_version=version
        )


logger = logging.getLogger(__name__)


class PDFConverter:
    """Gestionnaire de conversion HTML vers PDF"""
    
    DEFAULT_OPTIONS = {
        'page-size': 'A4',
        'margin-top': '0.5in',
        'margin-right': '0.5in',
        'margin-bottom': '0.5in',
        'margin-left': '0.5in',
        'encoding': 'UTF-8',
        'no-outline': None,
        'enable-local-file-access': None,
        'quiet': '',
        'print-media-type': None,
        'disable-smart-shrinking': None,
        'dpi': 300,
        'image-quality': 100,
        'enable-forms': None,
        'javascript-delay': 1000,
        'load-error-handling': 'ignore',
        'load-media-error-handling': 'ignore'
    }
    
    @staticmethod
    def is_available() -> bool:
        """Vérifie si la conversion PDF est disponible"""
        return PDFKIT_AVAILABLE
    
    @classmethod
    def convert(
        cls, 
        html_path: str, 
        pdf_path: str, 
        options: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Convertit HTML en PDF
        
        Returns:
            Tuple[bool, Optional[str]]: (succès, message d'erreur)
        """
        if not PDFKIT_AVAILABLE:
            return False, "pdfkit is not installed"
        
        if not os.path.exists(html_path):
            return False, f"HTML file not found: {html_path}"
        
        try:
            conversion_options = cls.DEFAULT_OPTIONS.copy()
            if options:
                conversion_options.update(options)
            
            pdfkit.from_file(html_path, pdf_path, options=conversion_options)
            
            if os.path.exists(pdf_path):
                return True, None
            else:
                return False, "PDF file was not created"
                
        except Exception as e:
            return False, f"Conversion error: {str(e)}"


class SummaryReportBuilder:
    """Constructeur de rapport de synthèse"""
    
    def __init__(self, version: str = '2.1'):
        self.version = version
        self.generation_time = datetime.now()

    def get_enhanced_css(self) -> str:
        """Retourne le CSS amélioré pour un PDF plus attrayant"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            margin: 0 auto;
            max-width: 800px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid #667eea;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        """

    def get_report_template(self) -> str:
        """Charge le template HTML depuis le fichier"""
        template_path = os.path.join(os.path.dirname(__file__), 'static', 'template.html')
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_css_path(self) -> str:
        """Retourne le chemin vers le fichier CSS"""
        return os.path.join(os.path.dirname(__file__), 'static', 'styles.css')

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


def bundle_outputs(file_paths: List[str], zip_path: str, top_level_dir: Optional[str] = None) -> None:
    """
    Crée un bundle des fichiers d'analyse
    
    Args:
        file_paths: Liste des chemins de fichiers à inclure
        zip_path: Chemin du fichier ZIP à créer
    """
    try:
        # Log incoming files and sizes
        valid_files = []
        total_size = 0
        for fp in file_paths:
            try:
                if os.path.exists(fp):
                    sz = os.path.getsize(fp)
                    valid_files.append(fp)
                    total_size += sz
                    logger.debug(f"bundle candidate: {fp} ({_format_size(sz)})")
                else:
                    logger.warning(f"bundle_outputs: file not found, skipping: {fp}")
            except Exception as e:
                logger.warning(f"bundle_outputs: error checking file {fp}: {e}")

        if not valid_files:
            raise FileNotFoundError("No valid files found to bundle")

        logger.info(f"Creating bundle {os.path.basename(zip_path)} with {len(valid_files)} files, total size={_format_size(total_size)}")

        # Choose compression strategy: for large total sizes prefer no compression to speed things up
        # Threshold can be tuned; here 20MB -> use store (no compression)
        size_threshold = 20 * 1024 * 1024
        if total_size >= size_threshold:
            compression = zipfile.ZIP_STORED
            compresslevel = None
            logger.info("Large bundle detected - using ZIP_STORED (no compression) for speed")
        else:
            compression = zipfile.ZIP_DEFLATED
            # use typical compression level
            compresslevel = 6

        # Create zip (allow Zip64 for large files)
        try:
            if compresslevel is not None:
                zf = zipfile.ZipFile(zip_path, 'w', compression=compression, compresslevel=compresslevel, allowZip64=True)
            else:
                zf = zipfile.ZipFile(zip_path, 'w', compression=compression, allowZip64=True)
        except TypeError:
            # Older Python versions may not support compresslevel argument
            zf = zipfile.ZipFile(zip_path, 'w', compression=compression, allowZip64=True)

        with zf:
            # Normalize top-level directory name
            if top_level_dir:
                top = top_level_dir.strip('/\\')
            else:
                top = 'analysis'

            for file_path in valid_files:
                arcname = os.path.join(top, os.path.basename(file_path))
                try:
                    zf.write(file_path, arcname)
                except Exception as e:
                    logger.warning(f"Failed adding {file_path} to zip: {e}")

            # Créer et ajouter le README inside top-level dir
            readme_content = _generate_readme(valid_files)
            zf.writestr(os.path.join(top, 'README.md'), readme_content)

            # Créer et ajouter les métadonnées
            metadata = AnalysisMetadata.from_files(valid_files)
            metadata_content = json.dumps(metadata.to_dict(), indent=2)
            zf.writestr(os.path.join(top, 'metadata.json'), metadata_content)

        # Final verification
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            raise RuntimeError("Zip file was not created or is empty")

        logger.info(f"Bundle created successfully: {os.path.basename(zip_path)} ({_format_size(os.path.getsize(zip_path))})")

    except Exception as e:
        logger.exception("Erreur lors de la création du bundle")
        raise


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