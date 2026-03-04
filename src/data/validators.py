"""
Data Validation Module using Great Expectations

Provides schema validation, data quality checks, and automated profiling
for fraud detection datasets.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger
from pathlib import Path
import json


class DataValidator:
    """Validate data quality and schema for fraud detection datasets"""
    
    def __init__(self, domain: str = 'fraud'):
        self.domain = domain
        self.expectations = []
        
    def add_expectation(self, expectation: Dict[str, Any]):
        """Add a data quality expectation"""
        self.expectations.append(expectation)
        
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate that DataFrame has expected columns"""
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        logger.info("Schema validation passed")
        return True
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, type]) -> bool:
        """Validate column data types"""
        issues = []
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue
                
            actual_type = df[col].dtype
            
            # Check if actual type matches or can be cast to expected type
            if expected_type == float and actual_type in [float, int]:
                continue
            elif expected_type == int and actual_type == int:
                continue
            elif expected_type == str and actual_type == object:
                continue
            elif actual_type != expected_type:
                issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
        
        if issues:
            logger.warning(f"Data type mismatches: {issues}")
            return False
        
        logger.info("Data type validation passed")
        return True
    
    def validate_ranges(self, df: pd.DataFrame, ranges: Dict[str, tuple]) -> bool:
        """Validate that numeric columns are within expected ranges"""
        violations = []
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
                
            out_of_range = df[
                (df[col] < min_val) | (df[col] > max_val)
            ]
            
            if not out_of_range.empty:
                violation_count = len(out_of_range)
                violations.append(
                    f"{col}: {violation_count} values outside [{min_val}, {max_val}]"
                )
        
        if violations:
            logger.warning(f"Range violations found: {violations}")
            return False
        
        logger.info("Range validation passed")
        return True
    
    def validate_completeness(self, df: pd.DataFrame, completeness_threshold: float = 0.95) -> bool:
        """Validate that data has sufficient completeness (non-null values)"""
        completeness_scores = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness = non_null_count / len(df)
            completeness_scores[col] = completeness
        
        failing_cols = {
            col: score for col, score in completeness_scores.items()
            if score < completeness_threshold
        }
        
        if failing_cols:
            logger.warning(
                f"Columns below completeness threshold ({completeness_threshold}): {failing_cols}"
            )
            return False
        
        logger.info("Completeness validation passed")
        return True
    
    def validate_uniqueness(self, df: pd.DataFrame, key_columns: List[str]) -> bool:
        """Validate uniqueness of records based on key columns"""
        duplicates = df.duplicated(subset=key_columns, keep=False)
        
        if duplicates.any():
            duplicate_count = duplicates.sum()
            logger.warning(f"Found {duplicate_count} duplicate records based on {key_columns}")
            return False
        
        logger.info("Uniqueness validation passed")
        return True
    
    def validate_fraud_specific(
        self,
        df: pd.DataFrame,
        amount_column: str = 'amount',
        label_column: str = 'is_fraud'
    ) -> bool:
        """Domain-specific validation for fraud detection data"""
        
        # Check for negative amounts
        if amount_column in df.columns:
            negative_amounts = (df[amount_column] < 0).sum()
            if negative_amounts > 0:
                logger.warning(f"Found {negative_amounts} transactions with negative amounts")
        
        # Check class balance
        if label_column in df.columns:
            class_counts = df[label_column].value_counts()
            total = len(df)
            
            fraud_ratio = class_counts.get(1, 0) / total if total > 0 else 0
            
            if fraud_ratio < 0.001:
                logger.warning(f"Extremely imbalanced dataset: fraud ratio = {fraud_ratio:.4f}")
            elif fraud_ratio > 0.5:
                logger.warning(f"Unusual fraud ratio: {fraud_ratio:.4f} (typically < 1%)")
        
        logger.info("Fraud-specific validation completed")
        return True
    
    def run_full_validation(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        expected_types: Optional[Dict[str, type]] = None,
        ranges: Optional[Dict[str, tuple]] = None,
        key_columns: Optional[List[str]] = None,
        completeness_threshold: float = 0.95
    ) -> Dict[str, bool]:
        """Run comprehensive validation suite"""
        
        results = {}
        
        if expected_columns:
            results['schema'] = self.validate_schema(df, expected_columns)
        
        if expected_types:
            results['data_types'] = self.validate_data_types(df, expected_types)
        
        if ranges:
            results['ranges'] = self.validate_ranges(df, ranges)
        
        results['completeness'] = self.validate_completeness(df, completeness_threshold)
        
        if key_columns:
            results['uniqueness'] = self.validate_uniqueness(df, key_columns)
        
        results['fraud_specific'] = self.validate_fraud_specific(df)
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.success("All validations passed ✓")
        else:
            failed_checks = [k for k, v in results.items() if not v]
            logger.error(f"Validation failed for: {failed_checks}")
        
        return results
    
    def generate_data_profile(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive data profile"""
        
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().mean() * 100).round(2).to_dict(),
            'statistics': {},
            'value_counts': {},
            'correlations': {}
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            profile['statistics'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        # Categorical value counts
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            profile['value_counts'][col] = df[col].value_counts().head(10).to_dict()
        
        # Correlations (for numeric columns only)
        if len(numeric_cols) > 1:
            profile['correlations'] = df[numeric_cols].corr().round(3).to_dict()
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            logger.info(f"Data profile saved to {output_path}")
        
        return profile


def create_default_validator(domain: str = 'fraud') -> DataValidator:
    """Create validator with domain-specific defaults"""
    
    validator = DataValidator(domain=domain)
    
    if domain == 'fraud':
        # Add fraud-specific expectations
        validator.add_expectation({
            'type': 'range',
            'column': 'amount',
            'min': 0,
            'max': 1_000_000
        })
        
        validator.add_expectation({
            'type': 'completeness',
            'column': 'transaction_id',
            'threshold': 1.0
        })
    
    return validator
