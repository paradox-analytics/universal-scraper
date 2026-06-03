"""
Test what Langchain Html2TextTransformer does to Metacritic HTML
"""
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document

# Read the cleaned HTML
with open("metacritic_cleaned.html", "r", encoding="utf-8") as f:
    html = f.read()

print(f"Original HTML: {len(html):,} bytes\n")

# Convert with Langchain (same as our DirectLLM extractor)
transformer = Html2TextTransformer()
doc = Document(page_content=html)
transformed_docs = transformer.transform_documents([doc])
text = transformed_docs[0].page_content

print(f"After Html2Text: {len(text):,} bytes\n")

# Show first 3000 characters
print("="*80)
print("FIRST 3000 CHARACTERS OF CONVERTED TEXT:")
print("="*80)
print(text[:3000])

# Save to file
with open("metacritic_converted.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("\n" + "="*80)
print(f"✓ Saved full converted text to: metacritic_converted.txt")

# Check if game names are still present
game_names = [
    "Zelda",
    "Tears of the Kingdom",
    "Elden Ring",
    "Baldur"
]

print("\n" + "="*80)
print("CHECKING IF GAME NAMES ARE PRESERVED:")
print("="*80)
for name in game_names:
    if name.lower() in text.lower():
        print(f"✓ Found: {name}")
    else:
        print(f"❌ Missing: {name}")

# Check if scores are preserved
print("\n" + "="*80)
print("CHECKING FOR SCORE PATTERNS:")
print("="*80)
import re
scores = re.findall(r'\b\d{2,3}\b', text[:5000])  # Look for 2-3 digit numbers in first 5000 chars
print(f"Found potential scores: {scores[:10]}")



