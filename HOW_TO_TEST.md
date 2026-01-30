# How to Test and Approve CSV Data

## 🎯 Goal

Test the new fixes and generate CSV files showing the corrected data extraction.

---

## 📋 Steps

### 1. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

(Replace `your-openai-api-key-here` with your actual key)

---

### 2. Run the Test Script

```bash
cd /Users/jevon_williams/Dev/universal-scraper
python3 test_and_generate_csvs.py
```

This will:
- Test Reddit, Apify, Metacritic, and eBay
- Generate new CSV files with "_FIXED" suffix
- Take about 5-10 minutes

---

### 3. Review the CSV Files

Compare the old vs new files:

**OLD (before fixes):**
- `reddit_sample.csv` → 4 app config items ❌
- `apify_sample.csv` → 2 JS libraries ❌
- `metacritic_sample.csv` → 5 GDPR configs ❌
- `ebay_sample.csv` → 33 UI actions ❌

**NEW (after fixes):**
- `reddit_sample_FIXED.csv` → Reddit posts ✅
- `apify_sample_FIXED.csv` → Apify actors ✅
- `metacritic_sample_FIXED.csv` → Video games ✅
- `ebay_sample_FIXED.csv` → Apple laptops ✅

---

## ✅ Approval Checklist

For each CSV file, verify:

### Reddit (`reddit_sample_FIXED.csv`)
- [ ] Has **title** field (post titles)
- [ ] Has **author** field (usernames)
- [ ] Has **upvotes** or **score** field
- [ ] Has **comments** count
- [ ] **NOT** app config like `ACCOUNT_MANAGER_ORIGIN`

### Apify (`apify_sample_FIXED.csv`)
- [ ] Has **name** field (actor names)
- [ ] Has **description** field
- [ ] Has **author** or **username** field
- [ ] **NOT** JS libraries like `Algolia Insights`

### Metacritic (`metacritic_sample_FIXED.csv`)
- [ ] Has **title** field (game names)
- [ ] Has **platform** field (PS5, Xbox, etc.)
- [ ] Has **metascore** or **rating** field
- [ ] Has **release_date** field
- [ ] **NOT** GDPR configs like `CPRA` or `Countries`

### eBay (`ebay_sample_FIXED.csv`)
- [ ] Has **title** or **name** field (laptop names)
- [ ] Has **price** field
- [ ] Has **condition** field
- [ ] Has **seller** or **seller_name** field
- [ ] **NOT** UI actions like `_type: Group` or `fieldId`

---

## 🔍 Quick CSV Inspection

You can quickly check the first few lines:

```bash
# Reddit
head -5 reddit_sample_FIXED.csv

# Apify
head -5 apify_sample_FIXED.csv

# Metacritic
head -5 metacritic_sample_FIXED.csv

# eBay
head -5 ebay_sample_FIXED.csv
```

---

## 📊 Expected Results

### Before Fixes:
- ❌ Reddit: 4 items (app config)
- ❌ Apify: 2 items (JS libraries)
- ❌ Metacritic: 5 items (GDPR config)
- ❌ eBay: 33 items (UI actions)
- **Total: 44 WRONG items**

### After Fixes:
- ✅ Reddit: 20-25 items (posts)
- ✅ Apify: 10-15 items (actors)
- ✅ Metacritic: 20-30 items (games)
- ✅ eBay: 50-60 items (laptops)
- **Total: 100-130 CORRECT items**

---

## 🐛 Troubleshooting

### "No OPENAI_API_KEY"
```bash
# Make sure you exported it in the same terminal
export OPENAI_API_KEY='your-key-here'

# Check it's set
echo $OPENAI_API_KEY
```

### "Module not found"
```bash
# Install dependencies
pip3 install -r requirements.txt
```

### "Browser not found"
```bash
# Install Playwright browsers
playwright install chromium
```

---

## 💡 Quick Test (Single Site)

If you want to test just ONE site quickly:

```bash
# Test only Reddit (faster)
python3 -c "
import asyncio
import os
from test_and_generate_csvs import test_and_save

async def quick_test():
    await test_and_save(
        'Reddit',
        'https://www.reddit.com/r/webscraping/',
        'Extract Reddit posts with title, author, upvotes',
        'reddit_quick_test.csv'
    )

asyncio.run(quick_test())
"
```

---

## 📝 Notes

- Each site takes 1-2 minutes to scrape
- The script shows progress for each site
- Browser windows will open/close automatically
- All generated CSV files are saved in the current directory

---

## ✨ After Approval

Once you've verified the CSV files are correct:

1. **Mark the fixes as approved** ✅
2. **Deploy to Apify** (if needed)
3. **Update documentation** with the new accuracy metrics
4. **Celebrate** 🎉 - the scraper now extracts the right data!

---

**Questions?** Check the detailed logs in the terminal output or review:
- `CSV_ANALYSIS_BEFORE_FIXES.md` - What was wrong
- `FIXES_COMPLETE.md` - What was fixed
- `IMPLEMENTATION_SUMMARY.md` - Summary for users








