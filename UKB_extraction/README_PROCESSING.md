# UK Biobank Respiratory Cohort Processing Pipeline

This pipeline extracts respiratory disease patients from UK Biobank and converts their data to natural language text for multimodal foundation model processing.

## Overview

The pipeline consists of two main components:

1. **A102_Explore-participant-data_Python.ipynb** - Data extraction and cohort filtering
2. **process_respiratory_cohort.py** - Natural language text conversion

## Workflow

### Step 1: Extract Data (A102 Notebook)

Run the notebook `A102_Explore-participant-data_Python.ipynb` on UK Biobank RAP Spark cluster to:

1. Extract all required fields from UK Biobank
2. Find all field instances (e.g., p50_i0, p50_i1, etc.)
3. Filter for respiratory disease patients (ICD-10: J09-J98, I26-I27)
4. Save results to CSV files

**Output files:**
- `ukb_respiratory_cohort.csv` - Filtered respiratory patients only
- `ukb_full_data.csv` - All extracted data
- `icd10_code_statistics.json` - Diagnosis code statistics

### Step 2: Convert to Natural Text

Run the Python script to convert structured data to natural language:

```bash
python process_respiratory_cohort.py \
    --input-csv ukb_respiratory_cohort.csv \
    --output-dir ./processed_patients \
    --max-patients 100  # Optional: limit number of patients
```

**Output:**
- Individual patient text files: `patient_{eid}.txt`
- JSON summary: `all_patients_processed.json`
- Statistics: `processing_statistics.json`

## Natural Text Format

Each patient's data is converted to natural language with sections:

### Example Output:

```
PATIENT ID: 1234567
================================================================================

DEMOGRAPHICS:
This male patient aged 65 years of White ethnicity has a BMI of 28.3 (overweight) 
with height 175.0 cm and weight 87.0 kg.

DIAGNOSES:
Respiratory diagnoses: chronic obstructive pulmonary disease (COPD) (J44), 
asthma (J45). Other diagnoses: hypertension (I10), type 2 diabetes (E11).

CLINICAL MEASUREMENTS:
Blood tests: White blood cell count: 7.2 × 10⁹/L; Hemoglobin concentration: 14.5 g/dL
Biochemistry tests: Albumin: 42.0 g/L; Creatinine: 85.0 µmol/L; Glucose: 5.8 mmol/L

ENVIRONMENTAL FACTORS:
Smoking status: previous smoker
Alcohol consumption: once or twice a week

RESPIRATORY ASSESSMENT:
Reports wheezing or whistling in the chest in the last year
Spirometry: FEV1: 2.35 liters, FVC: 3.80 liters

PSYCHOLOGICAL FACTORS:
Reports experiencing anxiety, tension or general nervousness
```

## Field Categories Included

The pipeline processes these UK Biobank field categories:

### Demographics
- Sex (p31)
- Birth year (p34)
- Ethnicity (p21000)
- BMI (p21001)
- Height (p50)
- Weight (p23104)

### Diagnoses
- ICD-10 hospital diagnoses (p41270, p41202, p41204, p41200)

### Clinical Measurements
- **Blood routine**: WBC, RBC, hemoglobin, hematocrit, platelets, etc. (p30000-p30150)
- **Biochemistry**: albumin, liver enzymes, glucose, cholesterol, etc. (p30600-p30890)

### Environmental Factors
- Smoking status (p20116)
- Alcohol intake (p1558)
- Physical activity (p22032)
- Sleep duration (p1160)

### Respiratory Assessment
- Wheeze/whistling (p2316)
- Shortness of breath (p4717)
- Chest pain (p2335)
- Spirometry: FEV1 (p3063), FVC (p3064)
- Peak expiratory flow (p3066)

### Psychological Factors
- Depression history (p4598)
- Anxiety (p5663)
- Neuroticism score (p20127)
- Loneliness (p2020)

## ICD-10 Code Mapping

The pipeline includes comprehensive ICD-10 code descriptions for:

- **J09-J98**: Respiratory system diseases
  - J09-J18: Influenza and pneumonia
  - J20-J22: Acute lower respiratory infections
  - J30-J39: Chronic upper respiratory diseases
  - J40-J47: Chronic lower respiratory diseases (COPD, asthma, etc.)
  - J60-J70: Lung diseases due to external agents
  - J80-J86: Respiratory conditions (ARDS, pulmonary edema, etc.)
  - J90-J94: Pleural conditions
  - J95-J99: Other respiratory disorders

- **I26-I27**: Pulmonary heart diseases
  - I26: Pulmonary embolism
  - I27: Other pulmonary heart diseases

## Data Quality

The script generates statistics on:
- Total patients processed
- Data completeness per field
- Demographic distributions
- Processing timestamps

## Requirements

### For A102 Notebook (DNAnexus RAP):
- Spark cluster instance (mem1_ssd1_v2_x8 recommended)
- dxdata package
- Python 3.9+

### For Processing Script:
```bash
pip install pandas numpy
```

## Usage Tips

1. **Test First**: Run with `--max-patients 10` to test the pipeline on a small subset

2. **Memory**: For large cohorts (>10,000 patients), process in batches:
   ```bash
   # Process first 5000
   python process_respiratory_cohort.py --input-csv data.csv --output-dir batch1 --max-patients 5000
   ```

3. **Data Validation**: Check `processing_statistics.json` to see data completeness

4. **Custom Fields**: Modify the field mappings in the script to add additional UK Biobank fields

## Troubleshooting

### Issue: Missing field data
**Solution**: Check `processing_statistics.json` to see which fields have low availability

### Issue: ICD-10 codes not recognized
**Solution**: The script uses the first 3 characters (e.g., J44 from J441). Add new codes to `icd10_map` dictionary

### Issue: Processing too slow
**Solution**: Use `--max-patients` to process in smaller batches

## Output for Foundation Models

The natural text output is ready for:
- Large Language Models (LLMs)
- Multimodal foundation models
- Clinical NLP pipelines
- Longitudinal patient trajectory analysis

Each patient's data is converted from structured database format to narrative clinical text, suitable for training or inference with medical foundation models.

## Next Steps

After processing:
1. Review sample patient files to validate text quality
2. Check statistics for data completeness
3. Use natural text files for downstream ML/AI tasks
4. Combine with imaging data if available for multimodal learning

## Citation

If using this pipeline, please cite:
- UK Biobank: https://www.ukbiobank.ac.uk/
- Your research paper/project

## Support

For issues or questions:
- Check UK Biobank documentation: https://biobank.ndph.ox.ac.uk/ukb/
- Review DNAnexus RAP documentation: https://dnanexus.gitbook.io/uk-biobank-rap/

