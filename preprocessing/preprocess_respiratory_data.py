"""
UK Biobank Respiratory Cohort - Data Preprocessing Pipeline

This script implements preprocessing steps similar to the UKB-MDRMF Scan.R pipeline,
adapted for the respiratory disease prediction dataset.

Features:
- Structured data preprocessing (missing values, outliers, normalization)
- Natural language text conversion for multimodal models (via natural_text_conversion.py)

Author: Adapted from UKB-MDRMF preprocessing pipeline
Date: 2025-01-08
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import natural text converter
from natural_text_conversion import NaturalTextConverter


class RespiratoryDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for UK Biobank respiratory cohort data.
    
    Pipeline steps:
    1. Load data and field mappings
    2. Handle special UK Biobank missing codes
    3. Remove features with high missing rates
    4. Split data into train/val/test
    5. Process continuous and categorical variables
    6. Optional: Detect and handle outliers (disabled by default)
    7. Optional: Normalize/standardize features (disabled by default)
    8. Impute missing values
    9. Generate natural language text (optional)
    10. Export processed data
    """
    
    def __init__(self, config):
        self.config = config
        self.field_lookup = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.processing_log = []
        self.df_original = None  # Store original data for natural text generation
        self.text_converter = NaturalTextConverter()  # Initialize natural text converter
        
    def log(self, message):
        """Log processing steps"""
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        self.processing_log.append({
            'timestamp': pd.Timestamp.now(),
            'message': message
        })
    
    def load_data(self, data_path):
        """
        Step 1: Load the respiratory cohort data
        """
        self.log("=" * 60)
        self.log("STEP 1: Loading Data")
        self.log("=" * 60)
        
        self.df = pd.read_csv(data_path, low_memory=True)
        self.log(f"Loaded data: {self.df.shape[0]:,} patients, {self.df.shape[1]:,} features")
        
        # Store original data for natural text generation (before any transformations)
        self.df_original = self.df.copy()
        self.log(f"Saved copy of original data for natural text generation")
        
        # Separate patient IDs
        self.patient_ids = self.df['eid'].copy()
        self.log(f"Extracted {len(self.patient_ids):,} patient IDs")
        
        return self
    
    def load_field_mapping(self, mapping_path):
        """
        Load field descriptions from mapping file
        """
        self.log("\nLoading field mapping...")
        
        with open(mapping_path, 'r') as f:
            field_mapping = json.load(f)
        
        # Create a lookup dictionary for all fields with recursive traversal
        def extract_fields(data, field_lookup_dict):
            """
            Recursively traverse the nested category structure to extract field information.
            Field IDs are numeric strings (e.g., "22032") and contain metadata like 'name', 'category', etc.
            """
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if this is a field entry (has 'name' and 'category' keys)
                    if 'name' in value and 'category' in value:
                        # This is a field ID entry
                        field_lookup_dict[key] = {
                            'description': value.get('name', ''),
                            'category': value.get('category', ''),
                            'value_type': value.get('value_type', ''),
                            'instances': value.get('instances', 0),
                            'arrays': value.get('arrays', 0)
                        }
                    else:
                        # Continue recursing into nested categories
                        extract_fields(value, field_lookup_dict)
            return field_lookup_dict
        
        self.field_lookup = extract_fields(field_mapping, {})
        
        self.log(f"Loaded {len(self.field_lookup)} field descriptions")
        return self
    
    def handle_special_codes(self):
        """
        Step 2: Handle special UK Biobank missing value codes
        
        UK Biobank uses special negative codes:
        -1: Date uncertain/unknown
        -2: System missing
        -3: Prefer not to answer
        -7: None of the above
        -11: Unknown/Missing
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 2: Handling Special Missing Value Codes")
        self.log("=" * 60)
        
        special_codes = self.config.get('special_codes', [-1, -2, -3, -7, -11])
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        replacements_made = 0
        for code in special_codes:
            count = (self.df[numeric_cols] == code).sum().sum()
            if count > 0:
                self.log(f"Converting {count:,} occurrences of code {code} to NaN")
                self.df[numeric_cols] = self.df[numeric_cols].replace(code, np.nan)
                replacements_made += count
        
        self.log(f"Total special codes converted to NaN: {replacements_made:,}")
        return self
    
    def filter_high_missing_features(self):
        """
        Step 3: Remove features with high missing rates
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 3: Filtering Features by Missing Rate")
        self.log("=" * 60)
        
        threshold = self.config.get('missing_threshold', 50)
        
        # Calculate missing percentages
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        
        # Keep features below threshold (plus eid)
        features_to_keep = ['eid'] + [col for col in self.df.columns 
                                       if col != 'eid' and missing_pct[col] <= threshold]
        
        n_original = self.df.shape[1]
        self.df = self.df[features_to_keep]
        n_filtered = self.df.shape[1]
        
        self.log(f"Missing threshold: {threshold}%")
        self.log(f"Features before filtering: {n_original}")
        self.log(f"Features after filtering: {n_filtered}")
        self.log(f"Features removed: {n_original - n_filtered}")
        
        return self
    
    def train_val_test_split(self):
        """
        Step 4: Split data into train/validation/test sets
        
        Important: This must be done BEFORE any imputation or normalization
        to prevent data leakage.
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 4: Train/Validation/Test Split")
        self.log("=" * 60)
        
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        test_ratio = self.config.get('test_ratio', 0.15)
        random_seed = self.config.get('random_seed', 42)
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            np.arange(len(self.df)),
            test_size=(1 - train_ratio),
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_size),
            random_state=random_seed
        )
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        self.log(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        self.log(f"Train samples: {len(train_idx):,} ({len(train_idx)/len(self.df)*100:.1f}%)")
        self.log(f"Validation samples: {len(val_idx):,} ({len(val_idx)/len(self.df)*100:.1f}%)")
        self.log(f"Test samples: {len(test_idx):,} ({len(test_idx)/len(self.df)*100:.1f}%)")
        
        return self
    
    def detect_outliers(self, data, method='iqr', threshold=3):
        """
        Detect outliers using IQR or Z-score method
        
        Args:
            data: pandas Series
            method: 'iqr' or 'zscore'
            threshold: multiplier for IQR or Z-score threshold
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        else:  # zscore
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers = pd.Series(False, index=data.index)
            outliers[data.dropna().index] = z_scores > threshold
        
        return outliers
    
    def process_continuous_variable(self, col_name):
        """
        Step 5: Process continuous variables
        
        For each continuous variable:
        1. Detect outliers (on training set only) - optional
        2. Create binary indicator for outliers - optional
        3. Cap outliers or remove them - optional
        4. Normalize using training set statistics - optional
        """
        train_data = self.df.loc[self.train_idx, col_name]
        
        # Detect and handle outliers (optional)
        n_outliers = 0
        if self.config.get('handle_outliers', False):
            outlier_method = self.config.get('outlier_method', 'iqr')
            outlier_threshold = self.config.get('outlier_threshold', 3)
            
            outliers = self.detect_outliers(train_data, outlier_method, outlier_threshold)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                # Create outlier indicator column
                outlier_col_name = f"{col_name}_outlier"
                self.df[outlier_col_name] = 0
                self.df.loc[self.train_idx[outliers], outlier_col_name] = 1
                
                # Cap outliers at boundaries
                if outlier_method == 'iqr':
                    Q1 = train_data.quantile(0.25)
                    Q3 = train_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_threshold * IQR
                    upper_bound = Q3 + outlier_threshold * IQR
                    
                    self.df[col_name] = self.df[col_name].clip(lower=lower_bound, upper=upper_bound)
        
        # Normalize using training set statistics (optional)
        if self.config.get('normalize_continuous', False):
            mean_val = train_data.mean()
            std_val = train_data.std()
            
            if std_val > 0:
                self.df[col_name] = (self.df[col_name] - mean_val) / std_val
        
        return n_outliers
    
    def process_categorical_variable(self, col_name):
        """
        Step 6: Process categorical variables
        
        For categorical variables:
        1. Encode categories (on training set)
        2. Handle rare categories (< 1% frequency)
        3. Apply encoding to all sets
        
        Note: Diagnosis fields (ICD10 codes stored as string lists) are skipped
        """
        train_data = self.df.loc[self.train_idx, col_name]
        
        # Check if this is a diagnosis field with list-like strings
        # These need special parsing and can't be simply encoded
        sample_val = train_data.dropna().iloc[0] if len(train_data.dropna()) > 0 else None
        if sample_val and isinstance(sample_val, str) and sample_val.startswith('['):
            # This is likely an ICD10 diagnosis field stored as string list
            # Skip it for now - needs special parsing
            self.log(f"  Skipping {col_name} - contains ICD10 codes that need special parsing")
            # Drop this column as it needs custom processing
            self.df.drop(columns=[col_name], inplace=True)
            return 0, 0
        
        # Convert all non-null values to strings to handle mixed types
        # This prevents issues with mixed float/string columns
        non_null_mask = self.df[col_name].notna()
        if non_null_mask.any():
            self.df.loc[non_null_mask, col_name] = self.df.loc[non_null_mask, col_name].astype(str)
        
        # Calculate category frequencies on training set
        train_data = self.df.loc[self.train_idx, col_name]
        value_counts = train_data.value_counts()
        total_count = len(train_data.dropna())
        
        if total_count == 0:
            # No valid training data, drop this column
            self.log(f"  Skipping {col_name} - no valid training data")
            self.df.drop(columns=[col_name], inplace=True)
            return 0, 0
        
        # Identify rare categories (< 1%)
        rare_threshold = self.config.get('rare_category_threshold', 0.01)
        rare_categories = value_counts[value_counts / total_count < rare_threshold].index.tolist()
        
        if len(rare_categories) > 0:
            # Group rare categories as 'Other'
            self.df.loc[self.df[col_name].isin(rare_categories), col_name] = 'Other'
        
        # Label encoding
        le = LabelEncoder()
        
        # Fit on training set (after grouping rare categories)
        train_values = self.df.loc[self.train_idx, col_name].dropna()
        
        if len(train_values) == 0:
            # No valid training data after cleaning
            self.log(f"  Skipping {col_name} - no valid training data after cleaning")
            self.df.drop(columns=[col_name], inplace=True)
            return 0, 0
        
        le.fit(train_values)
        
        # Map unseen labels to known categories (vectorized approach)
        non_null_mask = self.df[col_name].notna()
        unique_vals = self.df.loc[non_null_mask, col_name].unique()
        unseen_vals = [v for v in unique_vals if v not in le.classes_]
        
        if len(unseen_vals) > 0:
            # Map unseen values to 'Other' or most common class
            replacement = 'Other' if 'Other' in le.classes_ else train_values.mode()[0]
            self.df[col_name] = self.df[col_name].replace(unseen_vals, replacement)
        
        # Now transform all data (vectorized)
        non_null_mask = self.df[col_name].notna()
        self.df.loc[non_null_mask, col_name] = le.transform(self.df.loc[non_null_mask, col_name].values)
        
        self.label_encoders[col_name] = le
        
        return len(le.classes_), len(rare_categories)
    
    def process_all_variables(self):
        """
        Step 7: Process all variables based on their types
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 5-6: Processing Variables")
        self.log("=" * 60)
        
        # Log processing options
        self.log(f"Normalization: {'ENABLED' if self.config.get('normalize_continuous', False) else 'DISABLED'}")
        self.log(f"Outlier handling: {'ENABLED' if self.config.get('handle_outliers', False) else 'DISABLED'}")
        
        # Exclude eid from processing
        feature_cols = [col for col in self.df.columns if col != 'eid']
        
        continuous_count = 0
        categorical_count = 0
        
        for col in feature_cols:
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            
            # Determine variable type
            if dtype == 'object' or unique_count < 10:
                # Categorical
                n_classes, n_rare = self.process_categorical_variable(col)
                categorical_count += 1
                self.log(f"  Categorical: {col} - {n_classes} classes, {n_rare} rare")
            else:
                # Continuous
                n_outliers = self.process_continuous_variable(col)
                continuous_count += 1
                if n_outliers > 0:
                    self.log(f"  Continuous: {col} - {n_outliers} outliers detected")
        
        self.log(f"\nProcessed {continuous_count} continuous variables")
        self.log(f"Processed {categorical_count} categorical variables")
        
        return self
    
    def impute_missing_values(self):
        """
        Step 8: Impute remaining missing values
        
        Strategy:
        - Continuous: median imputation (from training set)
        - Categorical: mode imputation (from training set)
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 7: Missing Value Imputation")
        self.log("=" * 60)
        
        feature_cols = [col for col in self.df.columns if col != 'eid']
        
        for col in feature_cols:
            missing_count = self.df[col].isnull().sum()
            
            if missing_count > 0:
                # Get training set statistics
                train_data = self.df.loc[self.train_idx, col]
                
                if self.df[col].dtype in ['float64', 'int64']:
                    # Use median for numeric
                    fill_value = train_data.median()
                else:
                    # Use mode for categorical
                    fill_value = train_data.mode()[0] if len(train_data.mode()) > 0 else 0
                
                self.df[col].fillna(fill_value, inplace=True)
                self.log(f"  Imputed {missing_count:,} values in {col}")
        
        return self
    
    def generate_natural_text(self, dataset_name='train', max_patients=None):
        """
        Generate natural language text descriptions for patients
        
        Args:
            dataset_name: Name of dataset ('train', 'val', 'test', or 'all')
            max_patients: Maximum number of patients to process (None = all)
        """
        self.log("\n" + "=" * 60)
        self.log(f"STEP 9: Generating Natural Language Text ({dataset_name})")
        self.log("=" * 60)
        
        # Check if original data is available
        if self.df_original is None:
            self.log("⚠️  Warning: Original data not available. Using processed data.")
            data_source = self.df
        else:
            self.log("✓ Using original data (before transformations)")
            data_source = self.df_original
        
        # Select dataset
        if dataset_name == 'train':
            indices = self.train_idx
        elif dataset_name == 'val':
            indices = self.val_idx
        elif dataset_name == 'test':
            indices = self.test_idx
        else:  # 'all'
            indices = np.arange(len(data_source))
        
        # Use the NaturalTextConverter to process the cohort
        n_processed = self.text_converter.process_cohort(
            data=data_source,
            indices=indices,
            output_dir=self.config['output_dir'],
            dataset_name=dataset_name,
            max_patients=max_patients
        )
        
        self.log(f"✓ Generated natural text for {n_processed:,} patients")
        
        return self
    
    def extract_field_id(self, col_name):
        """Extract the base field ID from a column name like p30000_i0"""
        if col_name == 'eid':
            return 'eid'
        # Handle format: p{field_id}_i{instance}_a{array}
        if col_name.startswith('p'):
            parts = col_name[1:].split('_')
            return parts[0]
        return None
    
    def get_field_info(self, col_name):
        """Get description and category for a field"""
        field_id = self.extract_field_id(col_name)
        if field_id == 'eid':
            return {'description': 'Patient ID', 'category': 'identifier'}
        if field_id and field_id in self.field_lookup:
            return self.field_lookup[field_id]
        return {'description': 'Unknown', 'category': 'unknown'}
    
    def generate_hierarchical_feature_report(self):
        """
        Generate a hierarchical report of kept features organized by category
        """
        self.log("\nGenerating hierarchical feature report...")
        
        output_dir = Path(self.config['output_dir'])
        
        # Get all feature columns (exclude eid)
        feature_cols = [col for col in self.df.columns if col != 'eid']
        
        # Build hierarchical structure
        hierarchy = {}
        
        for col in feature_cols:
            field_info = self.get_field_info(col)
            category = field_info.get('category', 'unknown')
            description = field_info.get('description', 'Unknown')
            
            # Parse category path (split by ⏵ or similar separator)
            if '⏵' in category:
                cat_parts = [c.strip() for c in category.split('⏵')]
            else:
                cat_parts = [category]
            
            # Build nested structure
            current = hierarchy
            for part in cat_parts:
                if part not in current:
                    current[part] = {'_fields': []}
                current = current[part]
            
            # Add field to this category
            current['_fields'].append({
                'column': col,
                'field_id': self.extract_field_id(col),
                'description': description
            })
        
        # Write hierarchical report
        output_file = output_dir / 'kept_features_hierarchical.txt'
        
        def write_hierarchy(f, node, indent=0):
            """Recursively write hierarchical structure"""
            for key in sorted(node.keys()):
                if key == '_fields':
                    # Write fields at this level
                    for field in sorted(node['_fields'], key=lambda x: x['column']):
                        f.write(f"{'  ' * indent}├─ {field['column']} (Field {field['field_id']}): {field['description']}\n")
                else:
                    # Write category
                    f.write(f"{'  ' * indent}{'└─ ' if indent > 0 else ''}{key}\n")
                    write_hierarchy(f, node[key], indent + 1)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("KEPT FEATURES - HIERARCHICAL VIEW\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Features: {len(feature_cols)}\n")
            f.write(f"Missing Threshold: {self.config.get('missing_threshold', 50)}%\n")
            f.write("=" * 80 + "\n\n")
            
            write_hierarchy(f, hierarchy)
        
        self.log(f"✓ Saved hierarchical feature report: {output_file}")
        
        return self
    
    def save_processed_data(self):
        """
        Step 9: Save processed datasets
        """
        self.log("\n" + "=" * 60)
        self.log("STEP 8: Saving Processed Data")
        self.log("=" * 60)
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save complete dataset
        self.df.to_csv(output_dir / 'processed_data_complete.csv', index=False)
        self.log(f"✓ Saved complete dataset: {output_dir / 'processed_data_complete.csv'}")
        
        # Save train/val/test splits
        train_df = self.df.iloc[self.train_idx]
        val_df = self.df.iloc[self.val_idx]
        test_df = self.df.iloc[self.test_idx]
        
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        self.log(f"✓ Saved train set: {output_dir / 'train.csv'} ({len(train_df):,} samples)")
        self.log(f"✓ Saved validation set: {output_dir / 'val.csv'} ({len(val_df):,} samples)")
        self.log(f"✓ Saved test set: {output_dir / 'test.csv'} ({len(test_df):,} samples)")
        
        # Generate hierarchical feature report
        self.generate_hierarchical_feature_report()
        
        # Save processing log
        log_df = pd.DataFrame(self.processing_log)
        log_df.to_csv(output_dir / 'preprocessing_log.csv', index=False)
        self.log(f"✓ Saved processing log: {output_dir / 'preprocessing_log.csv'}")
        
        return self
    
    def run_full_pipeline(self, data_path, mapping_path):
        """
        Run the complete preprocessing pipeline
        """
        start_time = pd.Timestamp.now()
        self.log("=" * 60)
        self.log("UK BIOBANK RESPIRATORY COHORT PREPROCESSING PIPELINE")
        self.log("=" * 60)
        
        # Run all preprocessing steps
        self.load_data(data_path) \
            .load_field_mapping(mapping_path) \
            .handle_special_codes() \
            .filter_high_missing_features() \
            .train_val_test_split() \
            .process_all_variables() \
            .impute_missing_values() \
            .save_processed_data()
        
        # Optionally generate natural language text
        if self.config.get('generate_natural_text', False):
            datasets_to_convert = self.config.get('natural_text_datasets', ['train'])
            max_patients = self.config.get('natural_text_max_patients', None)
            
            for dataset_name in datasets_to_convert:
                self.generate_natural_text(dataset_name, max_patients)
        
        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        
        self.log("\n" + "=" * 60)
        self.log(f"PIPELINE COMPLETE - Duration: {duration:.2f} seconds")
        self.log("=" * 60)
        
        return self


# Main execution
if __name__ == "__main__":
    try:
        # Configuration
        config = {
            # Data preprocessing settings
            'missing_threshold': 50,  # Remove features with >50% missing
            'special_codes': [-1, -2, -3, -7, -11],  # UK Biobank special codes
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_seed': 42,
            'rare_category_threshold': 0.01,  # Categories < 1% grouped as 'Other'
            'output_dir': 'data/preprocessed',
            
            # Normalization and outlier handling (disabled by default)
            'normalize_continuous': False,  # Set True to normalize continuous variables
            'handle_outliers': False,  # Set True to detect and cap outliers
            'outlier_method': 'iqr',  # 'iqr' or 'zscore' (only used if handle_outliers=True)
            'outlier_threshold': 3,  # IQR multiplier or Z-score threshold (only used if handle_outliers=True)
            
            # Natural text generation settings
            'generate_natural_text': True,  # Set to True to generate natural language descriptions
            'natural_text_datasets': ['train'],  # Which datasets to convert: 'train', 'val', 'test', 'all'
            'natural_text_max_patients': 10,  # Max patients to convert (None = all). Set to small number for testing
        }
        
        # File paths
        data_path = 'data/ukb_respiratory_cohort_total.csv'
        mapping_path = 'UKB_extraction/field_mapping/ukb_field_mapping_new.json'
        
        # Run preprocessing
        preprocessor = RespiratoryDataPreprocessor(config)
        preprocessor.run_full_pipeline(data_path, mapping_path)
        
        print("\n✅ Preprocessing completed successfully!")
        print(f"📁 Output directory: {config['output_dir']}")
        
        if config.get('generate_natural_text', False):
            print("\n📝 Natural text generation enabled:")
            print(f"   - Datasets: {', '.join(config['natural_text_datasets'])}")
            if config.get('natural_text_max_patients'):
                print(f"   - Max patients per dataset: {config['natural_text_max_patients']}")
            else:
                print(f"   - Processing all patients")
            print(f"   - Text files saved in: {config['output_dir']}/natural_text_*/")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        raise

