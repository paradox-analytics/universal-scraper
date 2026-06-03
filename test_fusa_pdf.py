"""
Test PDF extraction on Fusa Nutrition authorization letter
"""

import asyncio
import json
import os
from universal_scraper.core.scraper import UniversalScraper


async def test_fusa_pdf():
    """Test with Fusa Nutrition authorization PDF"""
    
    print("\n" + "="*80)
    print("Testing PDF Extraction: Fusa Nutrition Authorization Letter")
    print("="*80)
    
    # Initialize scraper
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set")
        return
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_camoufox=False
    )
    
    # Try to find the PDF in common locations
    possible_paths = [
        "/Users/jevon_williams/Dropbox/Mac/Downloads/Fusa-Nutrition-LLC-99101569.pdf",
        "/Users/jevon_williams/Downloads/Fusa-Nutrition-LLC-99101569.pdf",
        "Fusa-Nutrition-LLC-99101569.pdf",
        "./Fusa-Nutrition-LLC-99101569.pdf"
    ]
    
    pdf_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pdf_path = path
            break
    
    if not pdf_path:
        print(f"\n❌ Could not find PDF in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return
    
    print(f"\n✅ Found PDF: {pdf_path}")
    print(f"   Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    # Extract with comprehensive fields for an authorization letter
    print("\n📄 Extracting ALL fields from PDF...")
    print("   Using comprehensive field list for authorization documents")
    
    pdf_url = f"file://{pdf_path}"
    
    result = await scraper.scrape(
        url=pdf_url,
        fields=[
            # Document info
            "document_type", "document_title", "date", "subject",
            
            # Trademark/Product info
            "product_name", "trademark", "uspto_number", "serial_number", "asin",
            
            # Rights Owner (Leoris Holdings)
            "rights_owner_company", "rights_owner_address", "rights_owner_phone", 
            "rights_owner_email", "rights_owner_city", "rights_owner_state", "rights_owner_zip",
            
            # Authorized Seller (Fusa Nutrition)
            "authorized_seller_company", "authorized_seller_address", 
            "authorized_seller_phone", "authorized_seller_email",
            "authorized_seller_city", "authorized_seller_state", "authorized_seller_zip",
            
            # Supplier
            "supplier_name", "authorized_supplier",
            
            # Authorization details
            "authorization_type", "platform", "marketplace",
            
            # Signatory
            "signatory_name", "signatory_title", "signatory_company"
        ]
    )
    
    print(f"\n✅ Extraction complete!")
    print(f"   Success: {result.get('success')}")
    print(f"   Source: {result.get('source')}")
    print(f"   Items extracted: {len(result.get('data', []))}")
    print(f"   Execution time: {result.get('metadata', {}).get('execution_time', 0):.2f}s")
    
    # Show results
    if result.get('data'):
        print(f"\n📊 Extracted Data:")
        print("="*80)
        
        for idx, item in enumerate(result['data'], 1):
            print(f"\n{'='*80}")
            print(f"ITEM {idx}")
            print('='*80)
            
            # Group fields by category
            categories = {
                "Document Info": ["document_type", "document_title", "date", "subject"],
                "Product/Trademark": ["product_name", "trademark", "uspto_number", "serial_number", "asin"],
                "Rights Owner": ["rights_owner_company", "rights_owner_address", "rights_owner_phone", 
                               "rights_owner_email", "rights_owner_city", "rights_owner_state", "rights_owner_zip"],
                "Authorized Seller": ["authorized_seller_company", "authorized_seller_address", 
                                     "authorized_seller_phone", "authorized_seller_email",
                                     "authorized_seller_city", "authorized_seller_state", "authorized_seller_zip"],
                "Supplier": ["supplier_name", "authorized_supplier"],
                "Authorization": ["authorization_type", "platform", "marketplace"],
                "Signatory": ["signatory_name", "signatory_title", "signatory_company"]
            }
            
            for category, fields in categories.items():
                category_data = {k: v for k, v in item.items() if k in fields and v}
                if category_data:
                    print(f"\n{category}:")
                    for key, value in category_data.items():
                        print(f"  {key}: {value}")
            
            # Show any other fields not in categories
            other_fields = {k: v for k, v in item.items() if k not in sum(categories.values(), []) and v}
            if other_fields:
                print(f"\nOther Fields:")
                for key, value in other_fields.items():
                    print(f"  {key}: {value}")
        
        # Save results
        output_file = "fusa_pdf_extraction.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to: {output_file}")
    
    elif result.get('error'):
        print(f"\n❌ Error: {result.get('error')}")
    
    await scraper.close()


async def main():
    await test_fusa_pdf()


if __name__ == "__main__":
    asyncio.run(main())







