"""
Example: Scraping a NEW Website (First Time)

This example demonstrates the 3 approaches to defining a schema
when scraping a website for the first time.
"""

import os
import sys
sys.path.insert(0, '.')

from universal_scraper.core.scraper import UniversalScraper
from universal_scraper.core.schema_inference import SchemaInference, infer_schema_from_scrape
from universal_scraper.core.schema_manager import SchemaDefinition, FieldMapping


# =============================================================================
# APPROACH 1: Quick Start (No Schema)
# =============================================================================
def approach_1_no_schema():
    """
    Quick start: Scrape without a schema
    
    Use this when:
    - You just want to test if the scraper works
    - You're exploring what data is available
    - You don't need schema stability yet
    
    Pros: ✅ Fastest to get started
    Cons: ❌ Output may change when website changes
    """
    print("\n" + "="*80)
    print("APPROACH 1: No Schema (Quick Start)")
    print("="*80)
    
    scraper = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid'
    )
    
    # Just specify fields you want
    fields = ['name', 'price', 'brand', 'description']
    
    result = scraper.scrape(
        'https://example.com/products',
        fields
    )
    
    print(f"✅ Scraped {len(result['data'])} items")
    print(f"📋 Fields extracted: {list(result['data'][0].keys())}")
    print("\n⚠️  Note: Output schema may change if website structure changes!")
    
    scraper.close()
    return result


# =============================================================================
# APPROACH 2: Auto-Generate Schema (Recommended for New Sites)
# =============================================================================
def approach_2_auto_generate():
    """
    Auto-generate schema from first scrape
    
    Use this when:
    - First time scraping a new website
    - You want schema stability going forward
    - You want the scraper to learn optimal mappings
    
    Pros: ✅ Automatic, ✅ Learns from actual data, ✅ Stable output
    Cons: ❌ Requires initial scrape to generate
    """
    print("\n" + "="*80)
    print("APPROACH 2: Auto-Generate Schema (Recommended)")
    print("="*80)
    
    # Step 1: Scrape without schema to learn
    print("\n📥 Step 1: Initial scrape to learn structure...")
    scraper = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid'
    )
    
    fields = ['name', 'price', 'brand', 'description', 'rating']
    result = scraper.scrape('https://example.com/products', fields)
    
    print(f"   Extracted {len(result['data'])} items")
    
    # Step 2: Auto-generate schema from the data
    print("\n🔍 Step 2: Analyzing data and generating schema...")
    inferencer = SchemaInference()
    inferencer.learn_from_data(result['data'])
    
    # Generate report to review what was learned
    report = inferencer.get_report()
    print(f"\n📊 Discovered Schema:")
    print(f"   Total fields: {report['fields_discovered']}")
    print(f"   Top fields:")
    for field in report['fields'][:5]:
        req = "REQUIRED" if field['required'] else "optional"
        print(f"     • {field['name']} → {field['normalized']} ({field['type']}, {field['coverage']}% coverage, {req})")
    
    # Step 3: Create schema
    print("\n🏗️  Step 3: Generating schema definition...")
    schema = inferencer.generate_schema(
        name="example_product",
        version="1.0",
        min_coverage=50.0  # Include fields present in 50%+ of items
    )
    
    print(f"   Schema: {schema.name} v{schema.version}")
    print(f"   Fields: {len(schema.fields)}")
    
    # Step 4: Save schema as code for reuse
    print("\n💾 Step 4: Exporting schema as Python code...")
    schema_code = inferencer.export_schema_code("example_product")
    
    with open('generated_schema_example_product.py', 'w') as f:
        f.write(schema_code)
    
    print("   ✅ Saved to: generated_schema_example_product.py")
    
    # Step 5: Use the schema for future scrapes
    print("\n🚀 Step 5: Using generated schema for future scrapes...")
    scraper_with_schema = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid',
        schema=schema,  # ← Now using the auto-generated schema
        strict_schema=False
    )
    
    result = scraper_with_schema.scrape('https://example.com/products', fields)
    quality = result['metadata']['schema_quality']
    
    print(f"   Schema Quality: {quality['status']} ({quality['success_rate']}% success)")
    print(f"   ✅ Output is now stable!")
    
    scraper.close()
    scraper_with_schema.close()
    
    return schema


# =============================================================================
# APPROACH 3: Manual Schema Definition (Full Control)
# =============================================================================
def approach_3_manual_schema():
    """
    Manually define schema upfront
    
    Use this when:
    - You know exactly what fields you need
    - You want full control over mappings
    - You have specific transformation requirements
    
    Pros: ✅ Full control, ✅ Optimal for specific use case
    Cons: ❌ Requires upfront knowledge of data structure
    """
    print("\n" + "="*80)
    print("APPROACH 3: Manual Schema Definition")
    print("="*80)
    
    # Define schema manually
    print("\n📋 Defining custom schema...")
    schema = SchemaDefinition(
        name="custom_product",
        version="1.0",
        fields=[
            FieldMapping(
                output_field="product_name",
                source_fields=["name", "title", "product_name", "productName"],
                field_type="string",
                required=True,
                aliases=["item_name"]
            ),
            FieldMapping(
                output_field="price_usd",
                source_fields=["price", "cost", "amount", "current_price"],
                field_type="number",
                required=True,
                transformer=lambda x: float(str(x).replace('$', '').replace(',', '')) if x else None
            ),
            FieldMapping(
                output_field="brand_name",
                source_fields=["brand", "manufacturer", "vendor"],
                field_type="string",
                required=False,
                transformer=lambda x: x.get('name') if isinstance(x, dict) else x
            ),
            FieldMapping(
                output_field="product_description",
                source_fields=["description", "details", "desc"],
                field_type="string",
                required=False
            ),
            FieldMapping(
                output_field="average_rating",
                source_fields=["rating", "average_rating", "stars"],
                field_type="number",
                required=False,
                transformer=lambda x: float(x) if x else None
            ),
        ]
    )
    
    print(f"   Schema: {schema.name} v{schema.version}")
    print(f"   Fields: {len(schema.fields)}")
    
    # Use the schema
    print("\n🚀 Scraping with custom schema...")
    scraper = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid',
        schema=schema,
        strict_schema=False
    )
    
    fields = ['name', 'price', 'brand', 'description', 'rating']
    result = scraper.scrape('https://example.com/products', fields)
    
    quality = result['metadata']['schema_quality']
    print(f"   Schema Quality: {quality['status']} ({quality['success_rate']}% success)")
    
    print("\n📦 Output Fields (Stable):")
    if result['data']:
        for field in result['data'][0].keys():
            print(f"     • {field}")
    
    scraper.close()
    return schema


# =============================================================================
# APPROACH 4: One-Liner Auto-Generation
# =============================================================================
def approach_4_one_liner():
    """
    One-liner: Infer schema from scrape
    
    Use this when:
    - You want the simplest possible workflow
    - You're okay with defaults
    
    Pros: ✅ Simplest, ✅ One function call
    Cons: ❌ Less control
    """
    print("\n" + "="*80)
    print("APPROACH 4: One-Liner Auto-Generation")
    print("="*80)
    
    # Step 1: Create scraper without schema
    scraper = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid'
    )
    
    # Step 2: Auto-generate schema from one scrape
    print("\n🔍 Auto-generating schema from first scrape...")
    schema = infer_schema_from_scrape(
        scraper=scraper,
        url='https://example.com/products',
        fields=['name', 'price', 'brand', 'description'],
        schema_name="auto_product",
        num_samples=1  # Learn from 1 scrape
    )
    
    print(f"\n✅ Schema generated: {schema.name}")
    print(f"   Fields: {len(schema.fields)}")
    
    # Step 3: Use it immediately
    print("\n🚀 Using generated schema...")
    scraper_with_schema = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid',
        schema=schema
    )
    
    result = scraper_with_schema.scrape(
        'https://example.com/products',
        ['name', 'price', 'brand']
    )
    
    print(f"   ✅ Scraped {len(result['data'])} items with stable schema")
    
    scraper.close()
    scraper_with_schema.close()
    return schema


# =============================================================================
# REAL EXAMPLE: Bootstrap Leafly Schema
# =============================================================================
def real_example_leafly():
    """
    Real example: Bootstrap schema for Leafly (new website)
    """
    print("\n" + "="*80)
    print("REAL EXAMPLE: Bootstrap Schema for Leafly")
    print("="*80)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Step 1: First scrape (no schema)
    print("\n📥 Step 1: Initial scrape to discover structure...")
    scraper = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid'
    )
    
    url = 'https://www.leafly.com/dispensary-info/mammoth-holistics/menu'
    fields = ['product_name', 'price', 'thc_content', 'cbd_content', 'brand', 'strain_type']
    
    result = scraper.scrape(url, fields)
    print(f"   ✅ Extracted {len(result['data'])} items")
    
    # Step 2: Learn schema
    print("\n🔍 Step 2: Analyzing data structure...")
    inferencer = SchemaInference()
    inferencer.learn_from_data(result['data'])
    
    report = inferencer.get_report()
    print(f"\n📊 Learned Schema:")
    for field in report['fields'][:7]:
        print(f"   • {field['name']} ({field['type']}, {field['coverage']}% coverage)")
    
    # Step 3: Generate schema
    print("\n🏗️  Step 3: Generating schema...")
    schema = inferencer.generate_schema("leafly_menu", version="1.0")
    
    # Step 4: Export as code
    print("\n💾 Step 4: Exporting schema code...")
    schema_code = inferencer.export_schema_code("leafly_menu")
    
    with open('generated_schema_leafly.py', 'w') as f:
        f.write(schema_code)
    
    print("   ✅ Saved to: generated_schema_leafly.py")
    print("   → You can now import and use this schema in production!")
    
    # Step 5: Test with schema
    print("\n🚀 Step 5: Testing with generated schema...")
    scraper_with_schema = UniversalScraper(
        model_name='gpt-4o-mini',
        fetch_mode='hybrid',
        schema=schema,
        strict_schema=False
    )
    
    result = scraper_with_schema.scrape(url, fields)
    quality = result['metadata']['schema_quality']
    
    print(f"\n✅ Schema Quality: {quality['status']} ({quality['success_rate']}% success)")
    print("\n📦 Stable Output Fields:")
    if result['data']:
        for field in result['data'][0].keys():
            print(f"   • {field}")
    
    scraper.close()
    scraper_with_schema.close()
    
    print("\n" + "="*80)
    print("🎯 RESULT: Schema is now stable for production use!")
    print("="*80)


# =============================================================================
# COMPARISON: Show All Approaches
# =============================================================================
def compare_all_approaches():
    """Compare all approaches side-by-side"""
    print("\n" + "="*80)
    print("📊 COMPARISON: All Approaches")
    print("="*80)
    
    print("""
┌────────────────────────────────────────────────────────────────────────────┐
│ APPROACH              │ SPEED    │ STABILITY │ CONTROL │ BEST FOR         │
├────────────────────────────────────────────────────────────────────────────┤
│ 1. No Schema          │ ⚡⚡⚡    │ ⚠️        │ 🎯      │ Quick tests      │
│ 2. Auto-Generate      │ ⚡⚡      │ ✅✅✅    │ 🎯🎯    │ New sites        │
│ 3. Manual Definition  │ ⚡        │ ✅✅✅    │ 🎯🎯🎯  │ Production       │
│ 4. One-Liner          │ ⚡⚡      │ ✅✅      │ 🎯      │ Rapid prototyping│
└────────────────────────────────────────────────────────────────────────────┘

RECOMMENDATION FOR NEW WEBSITES:
1. Start with Approach 2 (Auto-Generate)
2. Review the generated schema
3. Refine manually if needed
4. Export and commit to version control
    """)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("🚀 New Website Bootstrap Examples")
    print("="*80)
    
    print("\nChoose an example:")
    print("1. No Schema (Quick Start)")
    print("2. Auto-Generate Schema (Recommended)")
    print("3. Manual Schema Definition")
    print("4. One-Liner Auto-Generation")
    print("5. Real Example: Bootstrap Leafly")
    print("6. Compare All Approaches")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "1":
        approach_1_no_schema()
    elif choice == "2":
        approach_2_auto_generate()
    elif choice == "3":
        approach_3_manual_schema()
    elif choice == "4":
        approach_4_one_liner()
    elif choice == "5":
        real_example_leafly()
    elif choice == "6":
        compare_all_approaches()
    else:
        print("Invalid choice")
        compare_all_approaches()








