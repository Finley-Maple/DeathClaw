"""
Natural Language Text Conversion for UK Biobank Data

This module converts structured UK Biobank patient data into human-readable
clinical narratives suitable for LLMs and multimodal foundation models.

Two modes:
  1. Full narrative (original): broad clinical text across all UKB fields.
  2. Disease-before-60 narrative (new): restricted to disease diagnoses
     before age 60 plus key demographics, keyed by eid, for the survival
     cohort. Used by embedding method 1.

Author: Disease Prediction Research Team
Date: 2025-11-09
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import pandas as pd
import numpy as np
import re
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NaturalTextConverter:
    """
    Converts structured UK Biobank patient data to natural language text.
    
    Features:
    - Handles UK Biobank field name suffixes (_i0, _a0, _i0_a0, etc.)
    - Generates structured clinical narratives with 6 sections
    - Translates ICD-10 codes to human-readable descriptions
    - Formats clinical measurements with proper units
    """
    
    def __init__(self):
        """Initialize the converter with ICD-10 mappings"""
        
        # ICD-10 code descriptions for respiratory diseases
        self.icd10_map = {
            'J09': 'influenza due to pandemic virus',
            'J10': 'influenza due to seasonal virus',
            'J11': 'influenza unspecified',
            'J12': 'viral pneumonia',
            'J13': 'pneumonia due to Streptococcus pneumoniae',
            'J14': 'pneumonia due to Haemophilus influenzae',
            'J15': 'bacterial pneumonia',
            'J16': 'pneumonia due to other infectious organisms',
            'J17': 'pneumonia in diseases classified elsewhere',
            'J18': 'pneumonia unspecified organism',
            'J20': 'acute bronchitis',
            'J21': 'acute bronchiolitis',
            'J22': 'unspecified acute lower respiratory infection',
            'J30': 'allergic rhinitis',
            'J31': 'chronic rhinitis',
            'J32': 'chronic sinusitis',
            'J40': 'bronchitis not specified',
            'J41': 'simple chronic bronchitis',
            'J42': 'unspecified chronic bronchitis',
            'J43': 'emphysema',
            'J44': 'chronic obstructive pulmonary disease (COPD)',
            'J45': 'asthma',
            'J46': 'status asthmaticus',
            'J47': 'bronchiectasis',
            'J60': 'coalworker pneumoconiosis',
            'J61': 'pneumoconiosis due to asbestos',
            'J62': 'pneumoconiosis due to dust',
            'J63': 'pneumoconiosis due to other inorganic dusts',
            'J64': 'unspecified pneumoconiosis',
            'J66': 'airway disease due to specific organic dust',
            'J67': 'hypersensitivity pneumonitis',
            'J68': 'respiratory conditions due to inhalation of chemicals',
            'J69': 'pneumonitis due to solids and liquids',
            'J70': 'respiratory conditions due to external agents',
            'J80': 'acute respiratory distress syndrome',
            'J81': 'pulmonary edema',
            'J82': 'pulmonary eosinophilia',
            'J84': 'interstitial pulmonary diseases',
            'J85': 'abscess of lung and mediastinum',
            'J86': 'pyothorax',
            'J90': 'pleural effusion',
            'J91': 'pleural effusion in conditions',
            'J92': 'pleural plaque',
            'J93': 'pneumothorax',
            'J94': 'other pleural conditions',
            'J95': 'postprocedural respiratory disorders',
            'J96': 'respiratory failure',
            'J98': 'other respiratory disorders',
            'I26': 'pulmonary embolism',
            'I27': 'other pulmonary heart diseases'
        }
    
    def _get_field_value(self, patient_data, field_id):
        """
        Get field value handling UK Biobank suffixes (_i0, _i1, _a0, _i0_a0, etc.)
        
        Args:
            patient_data: Patient row (pandas Series)
            field_id: Base field ID (e.g., 'p30000')
            
        Returns:
            Field value or None if not found
        """
        # Try exact match first
        if field_id in patient_data.index and pd.notna(patient_data[field_id]):
            return patient_data[field_id]
        
        # Try with common UK Biobank suffixes (simple and compound)
        suffixes = [
            '_i0', '_i1', '_i2', '_i3',  # Instance suffixes
            '_a0', '_a1', '_a2', '_a3',  # Array suffixes
            '_i0_a0', '_i0_a1', '_i0_a2',  # Compound: instance 0 + array
            '_i1_a0', '_i1_a1', '_i1_a2',  # Compound: instance 1 + array
        ]
        for suffix in suffixes:
            field_with_suffix = f"{field_id}{suffix}"
            if field_with_suffix in patient_data.index and pd.notna(patient_data[field_with_suffix]):
                return patient_data[field_with_suffix]
        
        return None
    
    def convert_patient_to_text(self, patient_data: pd.Series) -> str:
        """
        Convert structured patient data to natural language text
        
        Args:
            patient_data: Single patient row from dataset
            
        Returns:
            Natural language description of patient data
        """
        text_sections = []
        
        # Add patient ID header
        patient_id = patient_data.get('eid', 'Unknown')
        text_sections.append(f"PATIENT ID: {patient_id}")
        text_sections.append("=" * 80)
        
        # Demographics section
        demo_text = self._create_demographics_text(patient_data)
        if demo_text:
            text_sections.append(f"\nDEMOGRAPHICS:\n{demo_text}")
        
        # Diagnoses section
        diag_text = self._create_diagnoses_text(patient_data)
        if diag_text:
            text_sections.append(f"\nDIAGNOSES:\n{diag_text}")
        
        # Clinical measurements section
        clinical_text = self._create_clinical_measurements_text(patient_data)
        if clinical_text:
            text_sections.append(f"\nCLINICAL MEASUREMENTS:\n{clinical_text}")
        
        # Environmental factors section
        env_text = self._create_environmental_text(patient_data)
        if env_text:
            text_sections.append(f"\nENVIRONMENTAL FACTORS:\n{env_text}")
        
        # Respiratory assessment section
        resp_text = self._create_respiratory_assessment_text(patient_data)
        if resp_text:
            text_sections.append(f"\nRESPIRATORY ASSESSMENT:\n{resp_text}")
        
        # Psychological factors section
        psych_text = self._create_psychological_text(patient_data)
        if psych_text:
            text_sections.append(f"\nPSYCHOLOGICAL FACTORS:\n{psych_text}")
        
        return "\n".join(text_sections)
    
    def _create_demographics_text(self, patient_data: pd.Series) -> str:
        """Create natural language text for demographics"""
        demo_parts = []
        
        # Sex
        sex_value = self._get_field_value(patient_data, 'p31')
        if sex_value is not None:
            if sex_value in [0, '0', 'Female']:
                sex = 'female'
            elif sex_value in [1, '1', 'Male']:
                sex = 'male'
            else:
                sex = 'unknown sex'
            demo_parts.append(f"This {sex} patient")
        
        # Age (calculate from birth year)
        birth_year_value = self._get_field_value(patient_data, 'p34')
        if birth_year_value is not None:
            try:
                birth_year = int(float(birth_year_value))
                current_year = datetime.now().year
                age = current_year - birth_year
                demo_parts.append(f"aged {age} years")
            except:
                pass
        
        # Ethnicity
        ethnicity_value = self._get_field_value(patient_data, 'p21000')
        if ethnicity_value is not None:
            ethnicity_map = {
                1: 'White', 1001: 'British', 1002: 'Irish', 1003: 'Any other white background',
                2: 'Mixed', 2001: 'White and Black Caribbean', 2002: 'White and Black African',
                2003: 'White and Asian', 2004: 'Any other mixed background',
                3: 'Asian', 3001: 'Indian', 3002: 'Pakistani', 3003: 'Bangladeshi', 3004: 'Any other Asian background',
                4: 'Black', 4001: 'Caribbean', 4002: 'African', 4003: 'Any other Black background',
                5: 'Chinese',
                6: 'Other ethnic group'
            }
            ethnicity_code = int(float(ethnicity_value))
            ethnicity = ethnicity_map.get(ethnicity_code, 'unknown ethnicity')
            demo_parts.append(f"of {ethnicity} ethnicity")
        
        # BMI
        bmi_value = self._get_field_value(patient_data, 'p21001')
        if bmi_value is not None:
            try:
                bmi = float(bmi_value)
                if bmi < 18.5:
                    bmi_category = 'underweight'
                elif bmi < 25:
                    bmi_category = 'normal weight'
                elif bmi < 30:
                    bmi_category = 'overweight'
                else:
                    bmi_category = 'obese'
                demo_parts.append(f"has a BMI of {bmi:.1f} ({bmi_category})")
            except:
                pass
        
        # Height
        height_value = self._get_field_value(patient_data, 'p50')
        if height_value is not None:
            try:
                height = float(height_value)
                demo_parts.append(f"with height {height:.1f} cm")
            except:
                pass
        
        # Weight
        weight_value = self._get_field_value(patient_data, 'p23104') # its BMI actually
        if weight_value is not None:
            try:
                BMI = float(weight_value)
                weight = BMI * height**2 / 10000
                demo_parts.append(f"and weight {weight:.1f} kg")
            except:
                pass
        
        return " ".join(demo_parts) + "." if demo_parts else ""
    
    def _create_diagnoses_text(self, patient_data: pd.Series) -> str:
        """Create natural language text for diagnoses"""
        diag_parts = []
        
        # ICD-10 diagnoses from multiple fields
        diag_fields = ['p41270', 'p41202', 'p41204', 'p41200']
        all_codes = []
        
        for field in diag_fields:
            field_value = self._get_field_value(patient_data, field)
            if field_value is not None:
                field_str = str(field_value)
                # Handle list format: ['J44', 'J45'] or just single codes
                if '[' in field_str:
                    # Parse list format
                    codes = re.findall(r"'([A-Z]\d+[^']*)'", field_str)
                    all_codes.extend(codes)
                else:
                    # Single code or comma-separated
                    codes = [c.strip() for c in field_str.split(',') if c.strip()]
                    all_codes.extend(codes)
        
        if all_codes:
            # Parse ICD-10 codes to human-readable descriptions
            readable_diagnoses = self._parse_icd10_codes(all_codes)
            
            # Separate respiratory and non-respiratory diagnoses
            respiratory_diag = []
            other_diag = []
            
            for desc, code in readable_diagnoses:
                if code.startswith('J') or code.startswith('I2'):
                    respiratory_diag.append(f"{desc} ({code})")
                else:
                    other_diag.append(f"{desc} ({code})")
            
            if respiratory_diag:
                diag_parts.append(f"Respiratory diagnoses: {', '.join(respiratory_diag)}")
            
            if other_diag:
                # Limit to top 10 other diagnoses to avoid clutter
                if len(other_diag) > 10:
                    diag_parts.append(f"Other diagnoses: {', '.join(other_diag[:10])} and {len(other_diag)-10} more")
                else:
                    diag_parts.append(f"Other diagnoses: {', '.join(other_diag)}")
        
        return ". ".join(diag_parts) + "." if diag_parts else "No diagnosis information available."
    
    def _parse_icd10_codes(self, codes: list) -> list:
        """Convert ICD-10 codes to human-readable descriptions"""
        readable = []
        seen_codes = set()
        
        for code in codes:
            if not code or code in seen_codes:
                continue
            
            code = code.strip().upper()
            seen_codes.add(code)
            
            # Extract the 3-character category (e.g., J44 from J441)
            match = re.match(r'([A-Z]\d{2})', code)
            if match:
                category = match.group(1)
                description = self.icd10_map.get(category, f"ICD-10 code {category}")
                readable.append((description, code))
            else:
                readable.append((f"ICD-10 code {code}", code))
        
        return readable
    
    def _create_clinical_measurements_text(self, patient_data: pd.Series) -> str:
        """Create natural language text for clinical measurements"""
        clinical_parts = []
        
        # Blood routine tests
        blood_tests = {
            'p30000': ('White blood cell count', '× 10⁹/L'),
            'p30010': ('Red blood cell count', '× 10¹²/L'),
            'p30020': ('Hemoglobin concentration', 'g/dL'),
            'p30030': ('Hematocrit percentage', '%'),
            'p30040': ('Mean corpuscular volume', 'fL'),
            'p30050': ('Mean corpuscular hemoglobin', 'pg'),
            'p30070': ('Red blood cell distribution width', '%'),
            'p30080': ('Platelet count', '× 10⁹/L'),
            'p30090': ('Platelet distribution width', ''),
            'p30100': ('Mean platelet volume', 'fL'),
            'p30120': ('Lymphocyte count', '× 10⁹/L'),
            'p30130': ('Monocyte count', '× 10⁹/L'),
            'p30140': ('Eosinophil count', '× 10⁹/L'),
            'p30150': ('Eosinophil count', '× 10⁹/L'),
        }
        
        blood_values = []
        for field, (name, unit) in blood_tests.items():
            value = self._get_field_value(patient_data, field)
            if value is not None:
                try:
                    value_float = float(value)
                    if unit:
                        blood_values.append(f"{name}: {value_float:.2f} {unit}")
                    else:
                        blood_values.append(f"{name}: {value_float:.2f}")
                except:
                    pass
        
        if blood_values:
            clinical_parts.append("Blood tests: " + "; ".join(blood_values))
        
        # Biochemistry tests
        biochem_tests = {
            'p30600': ('Albumin', 'g/L'),
            'p30610': ('Alkaline phosphatase', 'U/L'),
            'p30620': ('Alanine aminotransferase (ALT)', 'U/L'),
            'p30630': ('Apolipoprotein A', 'g/L'),
            'p30640': ('Apolipoprotein B', 'g/L'),
            'p30650': ('Aspartate aminotransferase (AST)', 'U/L'),
            'p30660': ('Direct bilirubin', 'µmol/L'),
            'p30670': ('Urea', 'mmol/L'),
            'p30680': ('Calcium', 'mmol/L'),
            'p30690': ('Cholesterol', 'mmol/L'),
            'p30700': ('Creatinine', 'µmol/L'),
            'p30710': ('C-reactive protein', 'mg/L'),
            'p30720': ('Cystatin C', 'mg/L'),
            'p30740': ('Glucose', 'mmol/L'),
            'p30750': ('Glycated hemoglobin (HbA1c)', 'mmol/mol'),
            'p30760': ('HDL cholesterol', 'mmol/L'),
            'p30770': ('IGF-1', 'nmol/L'),
            'p30780': ('LDL cholesterol', 'mmol/L'),
            'p30790': ('Lipoprotein A', 'nmol/L'),
            'p30810': ('Phosphate', 'mmol/L'),
            'p30860': ('Total protein', 'g/L'),
            'p30870': ('Triglycerides', 'mmol/L'),
            'p30880': ('Urate', 'µmol/L'),
        }
        
        biochem_values = []
        for field, (name, unit) in biochem_tests.items():
            value = self._get_field_value(patient_data, field)
            if value is not None:
                try:
                    value_float = float(value)
                    biochem_values.append(f"{name}: {value_float:.2f} {unit}")
                except:
                    pass
        
        if biochem_values:
            clinical_parts.append("Biochemistry tests: " + "; ".join(biochem_values))
        
        return "\n".join(clinical_parts) if clinical_parts else "No clinical measurement data available."
    
    def _create_environmental_text(self, patient_data: pd.Series) -> str:
        """Create natural language text for environmental factors"""
        env_parts = []
        
        # Smoking status
        smoking_value = self._get_field_value(patient_data, 'p20116')
        if smoking_value is not None:
            smoking_map = {
                0: 'never smoked',
                1: 'previous smoker',
                2: 'current smoker',
                -3: 'prefers not to answer'
            }
            smoking_code = int(float(smoking_value))
            smoking_status = smoking_map.get(smoking_code, 'unknown smoking status')
            env_parts.append(f"Smoking status: {smoking_status}")
        
        # Alcohol intake frequency
        alcohol_value = self._get_field_value(patient_data, 'p1558')
        if alcohol_value is not None:
            alcohol_map = {
                1: 'daily or almost daily',
                2: 'three or four times a week',
                3: 'once or twice a week',
                4: 'one to three times a month',
                5: 'special occasions only',
                6: 'never',
                -3: 'prefers not to answer'
            }
            alcohol_code = int(float(alcohol_value))
            alcohol_freq = alcohol_map.get(alcohol_code, 'unknown frequency')
            env_parts.append(f"Alcohol consumption: {alcohol_freq}")
        
        # Physical activity
        pace_value = self._get_field_value(patient_data, 'p22032')
        if pace_value is not None:
            try:
                walking_pace_map = {1: 'slow', 2: 'steady average', 3: 'brisk'}
                pace_code = int(float(pace_value))
                pace = walking_pace_map.get(pace_code, 'unknown pace')
                env_parts.append(f"Walking pace: {pace}")
            except:
                pass
        
        # Sleep duration
        sleep_value = self._get_field_value(patient_data, 'p1160')
        if sleep_value is not None:
            try:
                sleep_hours = float(sleep_value)
                env_parts.append(f"Sleep duration: {sleep_hours:.1f} hours per night")
            except:
                pass
        
        return "\n".join(env_parts) if env_parts else "No environmental factor data available."
    
    def _create_respiratory_assessment_text(self, patient_data: pd.Series) -> str:
        """Create natural language text for respiratory assessment"""
        resp_parts = []
        
        # Wheeze or whistling
        wheeze_value = self._get_field_value(patient_data, 'p2316')
        if wheeze_value is not None:
            if int(float(wheeze_value)) == 1:
                resp_parts.append("Reports wheezing or whistling in the chest in the last year")
        
        # Shortness of breath
        sob_value = self._get_field_value(patient_data, 'p4717')
        if sob_value is not None:
            if int(float(sob_value)) == 1:
                resp_parts.append("Experiences shortness of breath when walking on level ground at own pace")
        
        # Chest pain
        chest_pain_value = self._get_field_value(patient_data, 'p2335')
        if chest_pain_value is not None:
            if int(float(chest_pain_value)) == 1:
                resp_parts.append("Reports chest pain or discomfort")
        
        # Spirometry results
        spirometry_values = []
        fev1_value = self._get_field_value(patient_data, 'p3063')
        if fev1_value is not None:
            try:
                fev1 = float(fev1_value)
                spirometry_values.append(f"FEV1: {fev1:.2f} liters")
            except:
                pass
        
        fvc_value = self._get_field_value(patient_data, 'p3064')
        if fvc_value is not None:
            try:
                fvc = float(fvc_value)
                spirometry_values.append(f"FVC: {fvc:.2f} liters")
            except:
                pass
        
        if spirometry_values:
            resp_parts.append("Spirometry: " + ", ".join(spirometry_values))
        
        # Peak expiratory flow
        pef_value = self._get_field_value(patient_data, 'p3066')
        if pef_value is not None:
            try:
                pef = float(pef_value)
                resp_parts.append(f"Peak expiratory flow: {pef:.1f} L/min")
            except:
                pass
        
        return "\n".join(resp_parts) if resp_parts else "No respiratory assessment data available."
    
    def _create_psychological_text(self, patient_data: pd.Series) -> str:
        """Create natural language text for psychological factors"""
        psych_parts = []
        
        # Depression
        depression_value = self._get_field_value(patient_data, 'p4598')
        if depression_value is not None:
            if int(float(depression_value)) == 1:
                psych_parts.append("Has history of prolonged depression (lasting a whole week)")
        
        # Anxiety
        anxiety_value = self._get_field_value(patient_data, 'p5663')
        if anxiety_value is not None:
            if int(float(anxiety_value)) == 1:
                psych_parts.append("Reports experiencing anxiety, tension or general nervousness")
        
        # Neuroticism score
        neuroticism_value = self._get_field_value(patient_data, 'p20127')
        if neuroticism_value is not None:
            try:
                neuroticism = float(neuroticism_value)
                psych_parts.append(f"Neuroticism score: {neuroticism:.1f}")
            except:
                pass
        
        # Loneliness
        loneliness_value = self._get_field_value(patient_data, 'p2020')
        if loneliness_value is not None:
            loneliness_map = {1: 'yes', 0: 'no', -1: 'do not know', -3: 'prefer not to answer'}
            lonely_code = int(float(loneliness_value))
            lonely_status = loneliness_map.get(lonely_code)
            if lonely_status:
                psych_parts.append(f"Feels lonely or isolated: {lonely_status}")
        
        return "\n".join(psych_parts) if psych_parts else "No psychological factor data available."
    
    def process_cohort(self, data, indices, output_dir, dataset_name='train', max_patients=None):
        """
        Process a cohort of patients and generate natural text files
        
        Args:
            data: DataFrame with patient data
            indices: Indices of patients to process
            output_dir: Output directory path
            dataset_name: Name of dataset ('train', 'val', 'test')
            max_patients: Maximum number of patients to process
            
        Returns:
            Number of patients processed
        """
        # Limit number of patients if specified
        if max_patients and len(indices) > max_patients:
            indices = indices[:max_patients]
        
        # Create output directory
        output_path = Path(output_dir) / f'natural_text_{dataset_name}'
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Process each patient
        all_processed_data = []
        
        for idx, row_idx in enumerate(indices):
            try:
                # Get patient data
                patient_data = data.iloc[row_idx]
                patient_id = patient_data.get('eid', f'patient_{idx}')
                
                # Convert to natural text
                natural_text = self.convert_patient_to_text(patient_data)
                
                # Create output data structure
                processed_data = {
                    'patient_id': str(patient_id),
                    'natural_text': natural_text,
                    'processing_timestamp': datetime.now().isoformat(),
                    'dataset': dataset_name
                }
                
                all_processed_data.append(processed_data)
                
                # Save individual patient file
                patient_file = output_path / f'patient_{patient_id}.txt'
                with open(patient_file, 'w') as f:
                    f.write(natural_text)
                
            except Exception as e:
                print(f"  Error processing patient {idx}: {e}")
                continue
        
        # Save all processed data as JSON
        summary_file = output_path / 'all_patients_processed.json'
        with open(summary_file, 'w') as f:
            json.dump(all_processed_data, f, indent=2)
        
        return len(all_processed_data)


# ---------------------------------------------------------------------------
# Disease-before-60 text converter (new, for embedding method 1)
# ---------------------------------------------------------------------------

# Column name pattern in disease_trajectory.csv:
#   date_{icd_slug}_first_reported_{disease_name}_{field_id}_age
# e.g. date_e78_first_reported_disorders_of_lipoprotein_metabolism_and_other_lipidaemias_130636_age
import re

_COL_PATTERN = re.compile(
    r"^date_([a-z]\d+)_first_reported_(.+?)_(\d+)_age$"
)


def _parse_disease_col(col: str):
    """
    Parse a trajectory column name into (icd_code, readable_name).

    Example:
        'date_e78_first_reported_disorders_of_lipoprotein_metabolism_..._130636_age'
        → ('E78', 'disorders of lipoprotein metabolism and other lipidaemias')
    """
    m = _COL_PATTERN.match(col)
    if m:
        icd = m.group(1).upper()
        name = m.group(2).replace("_", " ")
        return icd, name
    return None, col.replace("_", " ")


class DiseaseBefore60TextConverter:
    """
    Produce a concise natural-language clinical summary per patient,
    with disease history **including age at diagnosis** before age 60.

    Uses the disease_trajectory.csv (age-at-diagnosis matrix) to produce
    text like:
        "At age 20.3, patient was diagnosed with chronic kidney disease.
         At age 40.6, patient was diagnosed with asthma."

    Input:
        - disease_trajectory.csv           (age at diagnosis per disease)
        - autoprognosis_survival_dataset.csv  (demographics / biomarkers)

    Output:
        - One text file per eid  OR  a single CSV  eid -> text
    """

    AGE_CUTOFF = 60.0

    SEX_MAP = {0: "female", 0.0: "female", 1: "male", 1.0: "male"}
    SMOKING_MAP = {0: "never smoker", 0.0: "never smoker",
                   1: "previous smoker", 1.0: "previous smoker",
                   2: "current smoker", 2.0: "current smoker"}
    ALCOHOL_MAP = {
        1: "daily or almost daily", 1.0: "daily or almost daily",
        2: "three or four times a week", 2.0: "three or four times a week",
        3: "once or twice a week", 3.0: "once or twice a week",
        4: "one to three times a month", 4.0: "one to three times a month",
        5: "special occasions only", 5.0: "special occasions only",
        6: "never", 6.0: "never",
    }

    def __init__(
        self,
        trajectory_csv: Path = None,
        survival_csv: Path = None,
    ):
        default_trajectory = PROJECT_ROOT / "data" / "preprocessed" / "disease_trajectory.csv"
        default_survival = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"

        self.trajectory_csv = trajectory_csv or default_trajectory
        self.survival_csv = survival_csv or default_survival

    def _load_data(self, eids: Optional[List[int]] = None):
        """Load trajectory matrix and survival demographics, merge by eid."""
        surv_df = pd.read_csv(self.survival_csv)
        surv_df["eid"] = surv_df["eid"].astype(int)

        traj_df = pd.read_csv(self.trajectory_csv)
        traj_df["eid"] = traj_df["eid"].astype(int)

        merged = surv_df.merge(traj_df, on="eid", how="inner", suffixes=("", "_traj"))
        # Drop duplicate columns
        dup_cols = [c for c in merged.columns if c.endswith("_traj")]
        merged.drop(columns=dup_cols, inplace=True)

        if eids is not None:
            merged = merged[merged["eid"].isin(set(eids))]

        # Identify age columns from trajectory
        self._age_columns = [c for c in traj_df.columns if c.endswith("_age") and c != "eid"]

        return merged

    def convert_row(self, row: pd.Series) -> str:
        """Convert a single patient row to a natural-language summary."""
        parts: List[str] = []

        # -- Demographics --------------------------------------------------
        eid = int(row.get("eid", 0))
        sex_raw = row.get("sex")
        sex = self.SEX_MAP.get(sex_raw, "unknown sex") if pd.notna(sex_raw) else "unknown sex"
        age_raw = row.get("age")
        birth_year = int(age_raw) if pd.notna(age_raw) else None

        demo = f"Patient {eid} is {sex}"
        if birth_year:
            demo += f", born in {birth_year}"
        parts.append(demo + ".")

        # BMI
        bmi = row.get("bmi")
        if pd.notna(bmi):
            cat = ("underweight" if bmi < 18.5 else "normal weight" if bmi < 25
                   else "overweight" if bmi < 30 else "obese")
            parts.append(f"BMI is {bmi:.1f} ({cat}).")

        # Smoking & alcohol
        sm = row.get("smoking_status")
        if pd.notna(sm):
            parts.append(f"Smoking status: {self.SMOKING_MAP.get(sm, 'unknown')}.")
        alc = row.get("alcohol_status")
        if pd.notna(alc):
            parts.append(f"Alcohol consumption: {self.ALCOHOL_MAP.get(alc, 'unknown')}.")

        # -- Disease history before 60 (with ages) -------------------------
        events: List[tuple] = []  # (age, "ICD_CODE disease_name")
        for col in self._age_columns:
            age_val = row.get(col)
            if pd.notna(age_val):
                age_years = float(age_val)
                if 0 <= age_years < self.AGE_CUTOFF:
                    icd, name = _parse_disease_col(col)
                    if icd:
                        events.append((age_years, f"{icd} {name}"))
                    else:
                        events.append((age_years, name))

        events.sort(key=lambda x: x[0])

        if events:
            disease_sentences = []
            for age, disease in events:
                disease_sentences.append(
                    f"At age {age:.1f}, patient was diagnosed with {disease}."
                )
            parts.append("Disease history before age 60: " + " ".join(disease_sentences))
        else:
            parts.append("No diseases diagnosed before age 60.")

        # -- Key biomarkers -----------------------------------------------
        biomarker_parts: List[str] = []
        bm_map = {
            "hdl_cholesterol": ("HDL cholesterol", "mmol/L"),
            "total_cholesterol": ("total cholesterol", "mmol/L"),
            "hba1c": ("HbA1c", "mmol/mol"),
            "c_reactive_protein": ("CRP", "mg/L"),
            "creatinine": ("creatinine", "µmol/L"),
            "haemoglobin": ("haemoglobin", "g/dL"),
        }
        for col, (name, unit) in bm_map.items():
            val = row.get(col)
            if pd.notna(val):
                biomarker_parts.append(f"{name} {float(val):.2f} {unit}")
        if biomarker_parts:
            parts.append("Key biomarkers: " + "; ".join(biomarker_parts) + ".")

        return " ".join(parts)

    def generate_texts(
        self,
        eids: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        output_csv: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Generate natural-language summaries for the given eids.

        Returns a DataFrame with columns [eid, text].
        Optionally writes individual .txt files and/or a CSV.
        """
        merged = self._load_data(eids)
        print(f"Building text for {len(merged)} patients "
              f"({len(self._age_columns)} disease age columns)...")
        records: List[Dict] = []

        for _, row in merged.iterrows():
            eid = int(row["eid"])
            text = self.convert_row(row)
            records.append({"eid": eid, "text": text})

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"eid_{eid}.txt").write_text(text, encoding="utf-8")

        df = pd.DataFrame(records)

        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Wrote {len(df)} patient texts to {output_csv}")

        return df


def main_before60():
    """CLI entry point for generating disease-before-60 texts."""
    parser = argparse.ArgumentParser(
        description="Generate natural-language disease-before-60 summaries per patient."
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "disease_trajectory.csv",
        help="Path to disease_trajectory.csv (age-at-diagnosis matrix).",
    )
    parser.add_argument(
        "--survival-csv",
        type=Path,
        default=PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv",
        help="Path to autoprognosis_survival_dataset.csv.",
    )
    parser.add_argument(
        "--cohort-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "cohort_split.json",
        help="Path to cohort_split.json (to restrict eids).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "text_before60",
        help="Directory to write individual eid_*.txt files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "text_before60.csv",
        help="CSV with columns [eid, text].",
    )
    args = parser.parse_args()

    # Load cohort eids
    eids = None
    if args.cohort_json.exists():
        with open(args.cohort_json) as f:
            cohort = json.load(f)
        eids = cohort["train_eids"] + cohort["val_eids"] + cohort["test_eids"]
        print(f"Loaded {len(eids)} eids from cohort split.")

    converter = DiseaseBefore60TextConverter(
        trajectory_csv=args.trajectory_csv,
        survival_csv=args.survival_csv,
    )
    converter.generate_texts(eids=eids, output_dir=args.output_dir, output_csv=args.output_csv)


if __name__ == "__main__":
    main_before60()

