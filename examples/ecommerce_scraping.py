"""
E-commerce Scraping Example
Extracting product data from e-commerce sites
"""

import os
import csv
from universal_scraper import UniversalScraper

API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    scraper = UniversalScraper(
        api_key=API_KEY,
        model_name='gpt-4o-mini'
    )
    
    # E-commerce fields
    fields = [
        'product_name',
        'product_price',
        'product_rating',
        'product_reviews_count',
        'product_availability',
        'product_image_url',
        'product_description'
    ]
    
    # Scrape product page
    result = scraper.scrape(
        url='https://books.toscrape.com/',
        fields=fields
    )
    
    print(f"\n✅ Extracted {len(result['data'])} products")
    
    # Save to CSV
    if result['data']:
        csv_file = 'products.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result['data'])
        
        print(f"💾 Saved to {csv_file}")
    
    # Display sample
    print("\n📦 Sample Products:")
    for i, product in enumerate(result['data'][:3], 1):
        print(f"\nProduct {i}:")
        for field, value in product.items():
            if value:
                print(f"  {field}: {value[:100] if isinstance(value, str) else value}")
    
    scraper.close()


if __name__ == '__main__':
    main()

