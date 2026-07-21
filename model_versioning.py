# model_versioning.py
"""
Model Versioning System for ForestGuard AI
Track, manage, and switch between different model versions
"""

import os
import json
import joblib
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersioning:
    """Manage multiple versions of ML models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.versions = []
        self.current_version = None
        self.load_versions()
    
    def load_versions(self):
        """Load all available model versions"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            return
        
        self.versions = []
        for item in os.listdir(self.models_dir):
            version_path = os.path.join(self.models_dir, item)
            if os.path.isdir(version_path) and item.startswith('v'):
                # Check if model files exist
                model_file = os.path.join(version_path, 'churn_model.pkl')
                metadata_file = os.path.join(version_path, 'metadata.pkl')
                
                if os.path.exists(model_file) and os.path.exists(metadata_file):
                    metadata = joblib.load(metadata_file)
                    self.versions.append({
                        'version': item,
                        'path': version_path,
                        'metadata': metadata,
                        'created_at': metadata.get('created_at', 'Unknown'),
                        'accuracy': metadata.get('accuracy', 0.0),
                        'roc_auc': metadata.get('roc_auc', 0.0)
                    })
        
        # Sort versions
        self.versions.sort(key=lambda x: x['version'], reverse=True)
        
        # Set current version
        if self.versions:
            self.current_version = self.versions[0]['version']
    
    def create_version(self, model, feature_names, metadata, version_name=None):
        """Create a new model version"""
        
        if version_name is None:
            # Auto-generate version name
            version_num = len(self.versions) + 1
            version_name = f"v{version_num}"
        
        version_dir = os.path.join(self.models_dir, version_name)
        os.makedirs(version_dir, exist_ok=True)
        
        # Add timestamp
        metadata['created_at'] = datetime.now().isoformat()
        metadata['version'] = version_name
        
        # Save model and artifacts
        joblib.dump(model, os.path.join(version_dir, 'churn_model.pkl'))
        joblib.dump(feature_names, os.path.join(version_dir, 'feature_names.pkl'))
        joblib.dump(metadata, os.path.join(version_dir, 'metadata.pkl'))
        
        # Save version info
        version_info = {
            'version': version_name,
            'created_at': metadata['created_at'],
            'accuracy': metadata.get('accuracy', 0.0),
            'roc_auc': metadata.get('roc_auc', 0.0),
            'description': metadata.get('description', '')
        }
        
        with open(os.path.join(version_dir, 'version_info.json'), 'w') as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Created model version: {version_name}")
        self.load_versions()
        
        return version_name
    
    def load_version(self, version):
        """Load a specific model version"""
        version_path = os.path.join(self.models_dir, version)
        
        if not os.path.exists(version_path):
            raise ValueError(f"Version {version} not found")
        
        model = joblib.load(os.path.join(version_path, 'churn_model.pkl'))
        feature_names = joblib.load(os.path.join(version_path, 'feature_names.pkl'))
        metadata = joblib.load(os.path.join(version_path, 'metadata.pkl'))
        
        return {
            'model': model,
            'feature_names': feature_names,
            'metadata': metadata,
            'version': version
        }
    
    def switch_version(self, version):
        """Switch to a different model version"""
        if version not in [v['version'] for v in self.versions]:
            raise ValueError(f"Version {version} not found")
        
        self.current_version = version
        logger.info(f"Switched to model version: {version}")
        
        # Save current version to config
        config = {
            'current_version': version,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.models_dir, 'current_version.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        return self.load_version(version)
    
    def compare_versions(self, version1, version2):
        """Compare two model versions"""
        v1 = self.load_version(version1)
        v2 = self.load_version(version2)
        
        comparison = {
            'version1': {
                'name': version1,
                'accuracy': v1['metadata'].get('accuracy', 0),
                'roc_auc': v1['metadata'].get('roc_auc', 0),
                'precision': v1['metadata'].get('precision', 0),
                'recall': v1['metadata'].get('recall', 0),
                'created_at': v1['metadata'].get('created_at', 'Unknown')
            },
            'version2': {
                'name': version2,
                'accuracy': v2['metadata'].get('accuracy', 0),
                'roc_auc': v2['metadata'].get('roc_auc', 0),
                'precision': v2['metadata'].get('precision', 0),
                'recall': v2['metadata'].get('recall', 0),
                'created_at': v2['metadata'].get('created_at', 'Unknown')
            }
        }
        
        # Calculate improvements
        comparison['improvements'] = {
            'accuracy': v2['metadata'].get('accuracy', 0) - v1['metadata'].get('accuracy', 0),
            'roc_auc': v2['metadata'].get('roc_auc', 0) - v1['metadata'].get('roc_auc', 0),
            'precision': v2['metadata'].get('precision', 0) - v1['metadata'].get('precision', 0),
            'recall': v2['metadata'].get('recall', 0) - v1['metadata'].get('recall', 0)
        }
        
        return comparison
    
    def delete_version(self, version):
        """Delete a model version"""
        if version == self.current_version:
            raise ValueError("Cannot delete current version")
        
        version_path = os.path.join(self.models_dir, version)
        if os.path.exists(version_path):
            shutil.rmtree(version_path)
            logger.info(f"Deleted model version: {version}")
            self.load_versions()
    
    def get_version_info(self, version=None):
        """Get information about a specific version"""
        if version is None:
            version = self.current_version
        
        version_info = [v for v in self.versions if v['version'] == version]
        if version_info:
            return version_info[0]
        return None
    
    def get_all_versions(self):
        """Get all available versions"""
        return self.versions
    
    def get_performance_history(self):
        """Get performance history across versions"""
        history = []
        for version in self.versions:
            history.append({
                'version': version['version'],
                'accuracy': version['accuracy'],
                'roc_auc': version['roc_auc'],
                'created_at': version['created_at']
            })
        return history
    
    def promote_version(self, version, new_name=None):
        """Promote a version to production"""
        if version not in [v['version'] for v in self.versions]:
            raise ValueError(f"Version {version} not found")
        
        # Create a symlink or copy to 'production' directory
        prod_dir = os.path.join(self.models_dir, 'production')
        if os.path.exists(prod_dir):
            shutil.rmtree(prod_dir)
        
        shutil.copytree(
            os.path.join(self.models_dir, version),
            prod_dir
        )
        
        logger.info(f"Promoted version {version} to production")
        
        # Update production metadata
        prod_metadata = {
            'production_version': version,
            'promoted_at': datetime.now().isoformat(),
            'promoted_by': 'system'
        }
        
        with open(os.path.join(self.models_dir, 'production_metadata.json'), 'w') as f:
            json.dump(prod_metadata, f, indent=2)
        
        return prod_dir

# Example usage
if __name__ == "__main__":
    # Initialize versioning system
    versioning = ModelVersioning()
    
    print("Available versions:", versioning.get_all_versions())
    print("Current version:", versioning.current_version)
    
    # Create a new version (example)
    # versioning.create_version(model, feature_names, metadata, "v2")