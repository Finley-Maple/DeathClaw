"""
UK Biobank Field Mapping Converter
Converts Excel-based field definitions to JSON format for data extraction pipeline

Date: 2025-01-09
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any


class FieldMappingConverter:
    """
    Converts UK Biobank field definitions from Excel format to JSON format.
    
    Expected Excel columns:
    - Field ID: UK Biobank field identifier (e.g., 21022, 41270)
    - Field Name: Human-readable field name
    - Category: Data category (Demographics, Clinical, Laboratory, etc.)
    - Value Type: continuous, categorical, date, text, etc.
    - Units: Measurement units (if applicable)
    - Description: Field description
    - Instances: Number of instances (visits)
    - Arrays: Number of array elements
    """
    
    def __init__(self):
        self.field_mapping = {}
        self.categories = {}
    
    @staticmethod
    def extract_category_level(category_str: str) -> str:
        """
        Extract the second-to-last level from a hierarchical category string.
        
        Args:
            category_str: Category string that may contain hierarchy (e.g., "A ⏵ B ⏵ C")
        
        Returns:
            - If multiple levels exist: second-to-last level (e.g., "B" from "A ⏵ B ⏵ C")
            - If single level: the category as-is
        
        Examples:
            "Additional exposures ⏵ Local environment ⏵ Residential noise pollution" → "Local environment"
            "Bone-densitometry of heel" → "Bone-densitometry of heel"
        """
        if pd.isna(category_str) or category_str == '':
            return 'General'
        
        # Split by the hierarchy separator
        parts = [part.strip() for part in str(category_str).split('⏵')]
        
        # If multiple levels, return second-to-last; otherwise return as-is
        if len(parts) >= 2:
            return parts[-2]
        else:
            return parts[0]
        
    def load_excel(self, excel_path: str, sheet_name: str = 'Sheet1') -> pd.DataFrame:
        """Load Excel file with field definitions"""
        print(f"Loading Excel file: {excel_path}")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"Loaded {len(df)} field definitions")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def convert_to_json(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert DataFrame to JSON structure matching original format.
        
        Output JSON structure:
        {
            "category_name": {
                "field_id": {
                    "name": "field_name",
                    "category": "category_name",
                    "value_type": "continuous/categorical/date/text",
                    "units": "units",
                    "description": "description",
                    "instances": 3,
                    "arrays": 0,
                    "coding": "coding_id" (optional)
                }
            }
        }
        """
        field_mapping = {}
        
        # Standardize column names (handle different naming conventions)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'field' in col_lower and 'id' in col_lower:
                column_map['field_id'] = col
            elif 'field' in col_lower and 'name' in col_lower:
                column_map['field_name'] = col
            # Check for value_type BEFORE category to avoid matching "Value Type" to category
            elif 'value' in col_lower and 'type' in col_lower:
                column_map['value_type'] = col
            elif 'category' in col_lower and 'id' not in col_lower:
                column_map['category'] = col
            elif 'unit' in col_lower:
                column_map['units'] = col
            elif 'description' in col_lower or 'desc' in col_lower:
                column_map['description'] = col
            elif 'instance' in col_lower:
                column_map['instances'] = col
            elif 'array' in col_lower:
                column_map['arrays'] = col
            elif 'coding' in col_lower:
                column_map['coding'] = col
        
        print(f"\nIdentified columns: {column_map}")
        
        # Convert each row to JSON entry, grouped by category
        for idx, row in df.iterrows():
            try:
                field_id = str(int(row[column_map.get('field_id', 'Field ID')]))
                
                # Extract category with hierarchical processing
                raw_category = row.get(column_map.get('category', 'Category'), 'General')
                extracted_category = self.extract_category_level(raw_category)
                
                field_entry = {
                    'name': str(row.get(column_map.get('field_name', 'Field Name'), '')),
                    'category': extracted_category,
                    'value_type': str(row.get(column_map.get('value_type', 'Value Type'), 'continuous')).lower(),
                    'description': str(row.get(column_map.get('description', 'Description'), ''))
                }
                
                # Add optional fields
                if column_map.get('units') in df.columns:
                    units = row.get(column_map['units'], '')
                    if pd.notna(units) and units != '':
                        field_entry['units'] = str(units)
                
                if column_map.get('instances') in df.columns:
                    instances = row.get(column_map['instances'], 1)
                    field_entry['instances'] = int(instances) if pd.notna(instances) else 1
                
                if column_map.get('arrays') in df.columns:
                    arrays = row.get(column_map['arrays'], 0)
                    field_entry['arrays'] = int(arrays) if pd.notna(arrays) else 0
                
                if column_map.get('coding') in df.columns:
                    coding = row.get(column_map['coding'], '')
                    if pd.notna(coding) and coding != '':
                        field_entry['coding'] = str(coding)
                
                # Group by category
                if extracted_category not in field_mapping:
                    field_mapping[extracted_category] = {}
                
                field_mapping[extracted_category][field_id] = field_entry
                
            except Exception as e:
                print(f"Warning: Skipping row {idx} due to error: {e}")
                continue
        
        # Count total fields across all categories
        total_fields = sum(len(fields) for fields in field_mapping.values())
        print(f"\nConverted {total_fields} fields to JSON format, grouped into {len(field_mapping)} categories")
        return field_mapping
    
    def categorize_fields(self, field_mapping: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group fields by category (already grouped in new structure)"""
        categories = {}
        for category, fields in field_mapping.items():
            categories[category] = list(fields.keys())
        
        return categories
    
    def save_json(self, field_mapping: Dict[str, Any], output_path: str):
        """Save field mapping to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(field_mapping, f, indent=2)
        print(f"\nSaved field mapping to: {output_path}")
        
    def generate_summary(self, field_mapping: Dict[str, Any]) -> str:
        """Generate summary statistics"""
        categories = self.categorize_fields(field_mapping)
        
        # Count total fields across all categories
        total_fields = sum(len(fields) for fields in field_mapping.values())
        
        summary = []
        summary.append("\n" + "="*80)
        summary.append("UK BIOBANK FIELD MAPPING SUMMARY")
        summary.append("="*80)
        summary.append(f"\nTotal Fields: {total_fields}")
        summary.append(f"\nFields by Category:")
        for category, fields in sorted(categories.items()):
            summary.append(f"  - {category}: {len(fields)} fields")
        
        # Count value types across all categories
        value_types = {}
        for category_fields in field_mapping.values():
            for field_info in category_fields.values():
                vtype = field_info.get('value_type', 'unknown')
                value_types[vtype] = value_types.get(vtype, 0) + 1
        
        summary.append(f"\nFields by Value Type:")
        for vtype, count in sorted(value_types.items()):
            summary.append(f"  - {vtype}: {count} fields")
        
        summary.append("\n" + "="*80)
        
        return "\n".join(summary)
    
    def convert(self, excel_path: str, json_output_path: str, sheet_name: str = 'Sheet1'):
        """
        Main conversion function.
        
        Args:
            excel_path: Path to Excel file with field definitions
            json_output_path: Path to save JSON output
            sheet_name: Excel sheet name (default: 'Sheet1')
        """
        # Load Excel
        df = self.load_excel(excel_path, sheet_name)
        
        # Convert to JSON
        field_mapping = self.convert_to_json(df)
        
        # Save JSON
        self.save_json(field_mapping, json_output_path)
        
        # Generate and print summary
        summary = self.generate_summary(field_mapping)
        print(summary)
        
        # Save summary
        summary_path = str(Path(json_output_path).with_suffix('.txt'))
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\nSaved summary to: {summary_path}")
        
        return field_mapping


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert UK Biobank field definitions from Excel to JSON')
    parser.add_argument('excel_file', type=str, help='Path to Excel file with field definitions')
    parser.add_argument('--output', '-o', type=str, default='ukb_field_mapping_new.json',
                        help='Output JSON file path')
    parser.add_argument('--sheet', '-s', type=str, default='Sheet1',
                        help='Excel sheet name')
    
    args = parser.parse_args()
    
    converter = FieldMappingConverter()
    converter.convert(args.excel_file, args.output, args.sheet)


if __name__ == '__main__':
    main()
