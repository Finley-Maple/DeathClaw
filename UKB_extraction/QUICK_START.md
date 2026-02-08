# Quick Start Guide: Respiratory Cohort to Natural Language Pipeline

## 🚀 Quick Overview

This pipeline extracts UK Biobank respiratory disease patients and converts their data to natural language text in **2 simple steps**.

---

## 📋 Prerequisites

1. **UK Biobank RAP Access** with a Spark cluster
2. **Python 3.9+** with pandas and numpy
3. **Field mapping file**: `ukb_field_mapping.json` or `field names.json`

---

## 🔄 Two-Step Workflow

### Step 1️⃣: Extract Data (Run on DNAnexus RAP)

**Open Jupyter on Spark Cluster** → Run `A102_Explore-participant-data_Python.ipynb`

```python
# The notebook will:
# 1. ✓ Extract all UK Biobank fields (with all instances)
# 2. ✓ Filter for respiratory patients (ICD-10: J09-J98, I26-I27)
# 3. ✓ Save to CSV files
```

**Expected Output:**
- ✅ `ukb_respiratory_cohort.csv` - Your filtered cohort
- ✅ `ukb_full_data.csv` - All data (backup)
- ✅ `icd10_code_statistics.json` - Statistics

---

### Step 2️⃣: Convert to Natural Text (Run anywhere)

Download the CSV file from RAP, then run:

```bash
python process_respiratory_cohort.py \
    --input-csv ukb_respiratory_cohort.csv \
    --output-dir ./processed_patients \
    --max-patients 100  # Optional: start with 100 to test
```

**Or run directly in the notebook** (last cells of A102):

```python
from process_respiratory_cohort import RespiratoryPatientTextConverter

converter = RespiratoryPatientTextConverter(
    input_csv='ukb_respiratory_cohort.csv',
    output_dir='./processed_patients'
)

converter.process_cohort(max_patients=10)
```

**Expected Output:**
- ✅ Individual text files: `patient_12345.txt`, `patient_67890.txt`, ...
- ✅ Summary: `all_patients_processed.json`
- ✅ Statistics: `processing_statistics.json`

---

## 📄 Example Natural Language Output

```
PATIENT ID: 1234567
================================================================================

DEMOGRAPHICS:
This male patient aged 65 years of White ethnicity has a BMI of 28.3 (overweight)
with height 175.0 cm and weight 87.0 kg.

DIAGNOSES:
Respiratory diagnoses: chronic obstructive pulmonary disease (COPD) (J440),
asthma (J45). Other diagnoses: hypertension (I10).

CLINICAL MEASUREMENTS:
Blood tests: White blood cell count: 7.2 × 10⁹/L; Hemoglobin: 14.5 g/dL
Biochemistry tests: Albumin: 42.0 g/L; Glucose: 5.8 mmol/L

ENVIRONMENTAL FACTORS:
Smoking status: previous smoker
Alcohol consumption: once or twice a week

RESPIRATORY ASSESSMENT:
Reports wheezing or whistling in the chest in the last year
Spirometry: FEV1: 2.35 liters, FVC: 3.80 liters
```

---

## 🎯 Key Features

### ✨ Smart Field Extraction
- **Automatic instance detection**: Finds all `p50_i0`, `p50_i1`, `p50_i2`, etc.
- **Fuzzy matching**: No need to specify every instance manually

### ✨ Accurate ICD-10 Filtering
- **Correct parsing**: Handles `J181` → `J18` (not `181`)
- **Respiratory codes**: J09-J98 (respiratory diseases) + I26-I27 (pulmonary heart)

### ✨ Rich Natural Text
- **6 categories**: Demographics, Diagnoses, Clinical, Environmental, Respiratory, Psychological
- **Human-readable**: ICD-10 codes → disease names
- **Context-aware**: Value ranges, units, clinical interpretations

---

## 🔧 Common Commands

### Test with small sample first:
```bash
python process_respiratory_cohort.py \
    --input-csv ukb_respiratory_cohort.csv \
    --output-dir ./test_output \
    --max-patients 5
```

### Process full cohort:
```bash
python process_respiratory_cohort.py \
    --input-csv ukb_respiratory_cohort.csv \
    --output-dir ./all_patients
```

### Check results:
```bash
# Count processed patients
ls -1 ./all_patients/patient_*.txt | wc -l

# View first patient
cat ./all_patients/patient_*.txt | head -n 50

# Check statistics
cat ./all_patients/processing_statistics.json | python -m json.tool
```

---

## 📊 What Data is Included?

| Category | Fields | Examples |
|----------|--------|----------|
| **Demographics** | 9 fields | Age, sex, ethnicity, BMI, height, weight |
| **Diagnoses** | ICD-10 codes | Hospital diagnoses from p41270, p41202, p41204 |
| **Blood Tests** | 15 fields | WBC, hemoglobin, platelets, lymphocytes, etc. |
| **Biochemistry** | 30+ fields | Albumin, glucose, cholesterol, liver enzymes |
| **Environmental** | 10+ fields | Smoking, alcohol, physical activity, sleep |
| **Respiratory** | 10+ fields | Wheeze, shortness of breath, FEV1, FVC, PEF |
| **Psychological** | 5+ fields | Depression, anxiety, neuroticism, loneliness |

---

## ⚠️ Troubleshooting

### Problem: "File not found"
**Solution**: Make sure CSV file is in the same directory, or use full path:
```bash
python process_respiratory_cohort.py \
    --input-csv /full/path/to/ukb_respiratory_cohort.csv \
    --output-dir ./output
```

### Problem: "Module not found"
**Solution**: Install dependencies:
```bash
pip install pandas numpy
```

### Problem: "Too slow"
**Solution**: Process in batches:
```bash
# First 1000 patients
python process_respiratory_cohort.py --input-csv data.csv --output-dir batch1 --max-patients 1000

# Next 1000 (you'd need to modify script or split CSV)
```

### Problem: "Many fields are 'No data available'"
**Solution**: This is normal! UK Biobank has selective data collection. Check `processing_statistics.json` to see field availability percentages.

---

## 📁 File Structure

```
UKB/
├── A102_Explore-participant-data_Python.ipynb    # Step 1: Extract data
├── process_respiratory_cohort.py                  # Step 2: Convert to text
├── README_PROCESSING.md                           # Detailed documentation
├── QUICK_START.md                                 # This file
│
├── ukb_respiratory_cohort.csv                     # Output from Step 1
├── ukb_full_data.csv                              # Full data backup
├── icd10_code_statistics.json                     # Diagnosis stats
│
└── processed_patients/                            # Output from Step 2
    ├── patient_1234567.txt
    ├── patient_7891011.txt
    ├── all_patients_processed.json
    └── processing_statistics.json
```

---

## 🎓 Next Steps

1. **Quality Check**: Review 5-10 sample patient files manually
2. **Statistics Review**: Check `processing_statistics.json` for data completeness
3. **Custom Fields**: Modify `process_respiratory_cohort.py` to add more fields
4. **Foundation Models**: Use the natural text for LLM training/inference
5. **Multimodal**: Combine with imaging data for complete patient profiles

---

## 💡 Tips

- **Start small**: Always test with `--max-patients 10` first
- **Check stats**: Review data completeness before full processing
- **Backup**: Keep the original CSV files
- **Version control**: Git track your field mapping changes
- **Documentation**: Note any custom modifications for reproducibility

---

## 📚 Related Files

- `README_PROCESSING.md` - Comprehensive documentation
- `A102_Explore-participant-data_Python.ipynb` - Main extraction notebook
- `process_respiratory_cohort.py` - Conversion script

---

## ✅ Success Criteria

You'll know it worked when you see:

```
INFO - Loading data from: ukb_respiratory_cohort.csv
INFO - Loaded 5421 patients with 156 fields
INFO - Processing patient 1/5421: 1234567
INFO - Processing patient 2/5421: 1234568
...
INFO - Processing complete! Processed 5421 patients
INFO - Individual patient files saved to: ./processed_patients
INFO - Summary file saved to: ./processed_patients/all_patients_processed.json
INFO - Statistics saved to: ./processed_patients/processing_statistics.json
```

---

**Questions?** Check `README_PROCESSING.md` for detailed documentation!

**Ready?** Start with Step 1 in the notebook! 🚀

