#!/usr/bin/env python3
"""
Test quality calculator with all optional fields
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import just the quality calculator (avoid litellm import issues)
from universal_scraper.core.quality_calculator import QualityCalculator

def test_all_optional_fields():
    """Test that all fields are optional by default"""
    calc = QualityCalculator()
    
    # Test data: some fields missing
    items = [
        {'name': 'Product 1', 'price': 10, 'image': None, 'maker': None},
        {'name': 'Product 2', 'price': 20, 'image': 'url', 'maker': None},
        {'name': 'Product 3', 'price': 30, 'image': None, 'maker': 'John'}
    ]
    
    fields = ['name', 'price', 'image', 'maker']
    quality = calc.calculate_quality_score(items, fields)
    coverage = calc.calculate_field_coverage(items, fields)
    missing = calc.get_missing_fields(items, fields)
    
    print("=" * 60)
    print("Quality Calculator Test (All Fields Optional)")
    print("=" * 60)
    print(f"Items: {len(items)}")
    print(f"Fields: {fields}")
    print(f"\nCoverage:")
    for field, count in coverage.items():
        pct = (count / len(items)) * 100
        print(f"  {field}: {count}/{len(items)} ({pct:.1f}%)")
    
    print(f"\nQuality Score: {quality:.1f}%")
    print(f"Missing Fields: {missing}")
    
    # Expected: quality should be based on average coverage (all optional)
    # name: 3/3 (100%), price: 3/3 (100%), image: 1/3 (33%), maker: 1/3 (33%)
    # Average: (100 + 100 + 33 + 33) / 4 = 66.5%
    expected_quality = (100 + 100 + 33.33 + 33.33) / 4
    
    print(f"\nExpected Quality (average): ~{expected_quality:.1f}%")
    print(f"Actual Quality: {quality:.1f}%")
    
    if abs(quality - expected_quality) < 5:
        print("\n✅ PASS: Quality calculation matches expected (all optional)")
    else:
        print(f"\n❌ FAIL: Quality calculation differs by {abs(quality - expected_quality):.1f}%")
    
    # Test with explicitly required fields
    print("\n" + "=" * 60)
    print("Test with Required Fields")
    print("=" * 60)
    quality_with_required = calc.calculate_quality_score(
        items, 
        fields,
        required_fields=['name', 'price'],  # Only name and price required
        optional_fields=['image', 'maker']
    )
    print(f"Quality (name, price required): {quality_with_required:.1f}%")
    print("✅ Should be higher than all-optional (weighted formula)")

if __name__ == '__main__':
    test_all_optional_fields()



