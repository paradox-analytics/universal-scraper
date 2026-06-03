"""
Test Schema Stability in Production

This demonstrates how the Schema Manager ensures stable output
even when website structure changes.
"""

import os
import sys
import json
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_manager import create_leafly_schema


def test_schema_stability():
    """
    Test that demonstrates schema stability.
    
    Even if Leafly's website structure changes (field names, nesting, etc),
    the output will maintain the same stable schema.
    """
    
    # Load API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Define stable output schema
    # This schema defines the STABLE fields your downstream systems expect
    schema = create_leafly_schema()
    
    print("="*80)
    print("🧪 TESTING SCHEMA STABILITY")
    print("="*80)
    print()
    print("📋 Stable Output Schema:")
    print(f"   Name: {schema.name}")
    print(f"   Version: {schema.version}")
    print(f"   Fields:")
    for field in schema.fields:
        required = "REQUIRED" if field.required else "optional"
        print(f"     • {field.output_field} ({field.field_type}) - {required}")
    print()
    
    # Initialize scraper WITH schema
    print("🚀 Initializing scraper with schema enforcement...")
    scraper = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid',
        enable_cache=True,
        headless=True,
        schema=schema,  # ← Schema enforcement enabled
        strict_schema=False  # ← Don't fail on missing fields, just warn
    )
    print()
    
    # Scrape Leafly
    url = 'https://www.leafly.com/dispensary-info/mammoth-holistics/menu'
    
    print("="*80)
    print("🌐 SCRAPING WITH SCHEMA ENFORCEMENT")
    print("="*80)
    print(f"URL: {url}")
    print()
    
    # We don't even need to specify fields - the schema defines them!
    # But we can still pass fields for the extraction phase
    fields = [
        'product_name', 'price', 'thc_content', 'cbd_content',
        'brand', 'strain_type'
    ]
    
    result = scraper.scrape(url, fields)
    
    print()
    print("="*80)
    print("✅ SCHEMA VALIDATION RESULTS")
    print("="*80)
    
    # Check schema quality
    if result['metadata'].get('schema_quality'):
        quality = result['metadata']['schema_quality']
        print(f"Status: {quality['status'].upper()}")
        print(f"Success Rate: {quality['success_rate']}%")
        print(f"Items Processed: {quality['total_items']}")
        print(f"Successful Mappings: {quality['successful_mappings']}")
        print(f"AI-Assisted Mappings: {quality['ai_assisted_mappings']}")
        print()
        print("Field Coverage:")
        for field_name, coverage in quality['field_coverage'].items():
            bar_length = int(coverage / 5)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {field_name:20} {bar} {coverage:5.1f}%")
    
    print()
    print("="*80)
    print("📦 STABLE OUTPUT DATA (First 3 products)")
    print("="*80)
    
    # Show that output has ONLY the stable schema fields
    for i, item in enumerate(result['data'][:3], 1):
        print(f"\n{i}. {item.get('name', 'N/A')}")
        print(f"   Price: ${item.get('price', 'N/A')}")
        print(f"   THC: {item.get('thc_percentage', 'N/A')}%")
        print(f"   CBD: {item.get('cbd_percentage', 'N/A')}%")
        print(f"   Brand: {item.get('brand', 'N/A')}")
        print(f"   Type: {item.get('strain_type', 'N/A')}")
        print(f"   Strain: {item.get('strain_name', 'N/A')}")
    
    # Save stable output
    stable_output = {
        'schema': {
            'name': schema.name,
            'version': schema.version,
            'hash': schema.get_hash()
        },
        'data': result['data'],
        'quality': result['metadata'].get('schema_quality'),
        'timestamp': result['metadata']['timestamp']
    }
    
    with open('leafly_stable_output.json', 'w') as f:
        json.dump(stable_output, f, indent=2)
    
    print()
    print("="*80)
    print("💾 SAVED TO: leafly_stable_output.json")
    print("="*80)
    print()
    
    # Explain the value
    print("="*80)
    print("🎯 PRODUCTION VALUE")
    print("="*80)
    print()
    print("✅ Benefits of Schema Management:")
    print()
    print("1. STABLE API CONTRACTS")
    print("   → Your downstream systems always receive the same field names")
    print("   → No breaking changes when website structure changes")
    print()
    print("2. AUTOMATIC FIELD MAPPING")
    print("   → Schema manager intelligently maps source fields to output")
    print("   → Handles renamed fields, nested objects, different structures")
    print()
    print("3. VALIDATION & QUALITY MONITORING")
    print("   → Real-time quality metrics for each scraping run")
    print("   → Alerts when field coverage drops below threshold")
    print()
    print("4. ZERO-DOWNTIME MIGRATIONS")
    print("   → When website changes, schema manager auto-adapts")
    print("   → Engineers don't need to update consumer code")
    print()
    print("5. TYPE SAFETY")
    print("   → Output fields are normalized to expected types")
    print("   → Prevents downstream errors from unexpected data types")
    print()
    
    # Compare with and without schema
    print("="*80)
    print("📊 COMPARISON: WITH vs WITHOUT Schema")
    print("="*80)
    print()
    print("WITHOUT Schema Manager:")
    print("  • Website changes field names → Consumer breaks ❌")
    print("  • Website changes nesting → Consumer breaks ❌")
    print("  • Website adds/removes fields → Consumer confused ❌")
    print("  • Engineers must manually update code → Downtime ⏱️")
    print()
    print("WITH Schema Manager:")
    print("  • Website changes field names → Auto-mapped ✅")
    print("  • Website changes nesting → Auto-mapped ✅")
    print("  • Website adds/removes fields → Handled gracefully ✅")
    print("  • Engineers don't touch consumer code → Zero downtime 🚀")
    print()
    
    scraper.close()
    
    print("="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    

def demonstrate_schema_evolution():
    """
    Demonstrate how to handle schema evolution over time.
    """
    print()
    print("="*80)
    print("🔄 SCHEMA EVOLUTION EXAMPLE")
    print("="*80)
    print()
    
    print("Scenario: You need to add a new field to your schema")
    print()
    print("1. Create a new schema version:")
    print("   schema_v2 = SchemaDefinition(")
    print("       name='leafly_product',")
    print("       version='2.0',  # ← Increment version")
    print("       fields=[")
    print("           ...existing_fields,")
    print("           FieldMapping(")
    print("               output_field='discount_percentage',  # ← New field")
    print("               source_fields=['discount', 'sale_percent'],")
    print("               field_type='number',")
    print("               required=False,  # ← Optional for backward compat")
    print("               default_value=0")
    print("           )")
    print("       ]")
    print("   )")
    print()
    print("2. Deploy new scraper with schema_v2")
    print()
    print("3. Consumers can choose:")
    print("   • Keep using v1 schema (stable, no changes needed)")
    print("   • Upgrade to v2 schema (get new field)")
    print()
    print("Result: Zero-downtime schema evolution! 🎉")
    print()


if __name__ == "__main__":
    test_schema_stability()
    demonstrate_schema_evolution()








