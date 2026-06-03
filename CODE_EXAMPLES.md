# Universal Scraper - Practical Code Examples

This document provides practical, production-ready code examples for common use cases.

---

## Example 1: E-Commerce Product Monitoring System

**Use Case**: Monitor competitor pricing across multiple e-commerce sites.

```python
"""
E-Commerce Price Monitoring System
Monitors product prices and sends alerts on changes
"""

import asyncio
import logging
from typing import List, Dict
from datetime import datetime
from universal_scraper import UniversalScraper
from universal_scraper.core import SchemaDefinition, SchemaField

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductMonitor:
    """Monitor product prices across multiple sites"""
    
    def __init__(self, api_key: str, alert_callback=None):
        # Define consistent product schema
        self.schema = SchemaDefinition(
            name="product_price",
            version="1.0",
            fields=[
                SchemaField("product_name", "string", required=True),
                SchemaField("price", "float", required=True),
                SchemaField("currency", "string", required=False),
                SchemaField("availability", "boolean", required=True),
                SchemaField("seller", "string", required=False),
            ]
        )
        
        # Initialize scraper with schema
        self.scraper = UniversalScraper(
            api_key=api_key,
            schema=self.schema,
            strict_schema=True,
            fetch_mode="hybrid",
            enable_cache=True
        )
        
        self.alert_callback = alert_callback
        self.price_history = {}
    
    async def monitor_products(
        self, 
        products: List[Dict[str, str]], 
        check_interval: int = 3600
    ):
        """
        Continuously monitor products
        
        Args:
            products: List of {'name': str, 'url': str}
            check_interval: Seconds between checks (default: 1 hour)
        """
        
        while True:
            logger.info(f"🔍 Checking {len(products)} products...")
            
            # Check all products in parallel
            tasks = [
                self.check_product(p['name'], p['url'])
                for p in products
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for product, result in zip(products, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error checking {product['name']}: {result}")
                elif result:
                    await self.process_price_change(product['name'], result)
            
            # Wait before next check
            logger.info(f"⏰ Next check in {check_interval}s")
            await asyncio.sleep(check_interval)
    
    async def check_product(self, product_name: str, url: str) -> Dict:
        """Check single product price"""
        
        try:
            result = await self.scraper.scrape(url, fields=[])
            
            if result['data']:
                product = result['data'][0]
                logger.info(
                    f"✅ {product_name}: ${product.get('price')} "
                    f"({result['metadata']['fetch_method']})"
                )
                return product
            else:
                logger.warning(f"⚠️ No data found for {product_name}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Failed to scrape {product_name}: {e}")
            raise
    
    async def process_price_change(self, product_name: str, current_data: Dict):
        """Process price changes and trigger alerts"""
        
        current_price = current_data.get('price')
        
        # Get historical price
        if product_name in self.price_history:
            previous_price = self.price_history[product_name]['price']
            
            # Calculate change
            if current_price != previous_price:
                change_pct = ((current_price - previous_price) / previous_price) * 100
                
                logger.info(
                    f"💰 Price change for {product_name}: "
                    f"${previous_price} → ${current_price} ({change_pct:+.2f}%)"
                )
                
                # Trigger alert
                if self.alert_callback:
                    await self.alert_callback(
                        product_name=product_name,
                        old_price=previous_price,
                        new_price=current_price,
                        change_pct=change_pct,
                        data=current_data
                    )
        
        # Update history
        self.price_history[product_name] = {
            'price': current_price,
            'timestamp': datetime.now().isoformat(),
            'data': current_data
        }
    
    async def close(self):
        """Clean up resources"""
        await self.scraper.close()


# Example usage
async def send_slack_alert(product_name, old_price, new_price, change_pct, data):
    """Send alert to Slack"""
    import aiohttp
    
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    message = {
        "text": f"🔔 Price Alert: {product_name}",
        "attachments": [{
            "color": "good" if change_pct < 0 else "danger",
            "fields": [
                {"title": "Old Price", "value": f"${old_price}", "short": True},
                {"title": "New Price", "value": f"${new_price}", "short": True},
                {"title": "Change", "value": f"{change_pct:+.2f}%", "short": True},
                {"title": "Available", "value": str(data.get('availability')), "short": True}
            ]
        }]
    }
    
    async with aiohttp.ClientSession() as session:
        await session.post(webhook_url, json=message)


async def main():
    # Products to monitor
    products = [
        {
            "name": "iPhone 15 Pro",
            "url": "https://amazon.com/dp/B0CHX1W1XY"
        },
        {
            "name": "Samsung Galaxy S24",
            "url": "https://amazon.com/dp/B0CMDWC5KL"
        },
        {
            "name": "Google Pixel 8",
            "url": "https://amazon.com/dp/B0CGT96SZ5"
        }
    ]
    
    # Initialize monitor
    monitor = ProductMonitor(
        api_key="your-openai-key",
        alert_callback=send_slack_alert
    )
    
    try:
        # Start monitoring (runs forever)
        await monitor.monitor_products(products, check_interval=3600)
    finally:
        await monitor.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Example 2: Real Estate Data Aggregator

**Use Case**: Aggregate real estate listings from multiple sources.

```python
"""
Real Estate Data Aggregator
Crawls multiple real estate sites and normalizes data
"""

import asyncio
from typing import List, Dict
from universal_scraper.orchestrator import UniversalWorkflow, WorkflowConfig, WorkflowMode
from universal_scraper.crawler import CrawlConfig
from universal_scraper.core import SchemaDefinition, SchemaField


class RealEstateAggregator:
    """Aggregate listings from multiple real estate sites"""
    
    def __init__(self, api_key: str):
        # Define standard real estate schema
        self.schema = SchemaDefinition(
            name="real_estate_listing",
            version="2.0",
            fields=[
                SchemaField("address", "string", required=True),
                SchemaField("price", "float", required=True),
                SchemaField("bedrooms", "integer", required=True),
                SchemaField("bathrooms", "float", required=True),
                SchemaField("square_feet", "integer", required=False),
                SchemaField("property_type", "string", required=False),
                SchemaField("listing_url", "string", required=True),
                SchemaField("agent_name", "string", required=False),
                SchemaField("agent_phone", "string", required=False),
                SchemaField("description", "string", required=False),
                SchemaField("images", "list", required=False)
            ]
        )
        
        # Configure crawler
        self.crawl_config = CrawlConfig(
            mode='smart',
            max_depth=3,
            max_pages=500,
            handle_pagination=True,
            discover_apis=True,
            url_patterns=[
                r'/property/\d+',
                r'/listing/[\w-]+',
                r'/homes/[\w-]+'
            ]
        )
        
        # Initialize workflow
        self.workflow = UniversalWorkflow(
            config=WorkflowConfig(
                mode=WorkflowMode.CRAWL_THEN_SCRAPE,
                crawl_config=self.crawl_config,
                schema=self.schema,
                strict_schema=False,  # Allow partial data
                fields=[
                    'address', 'price', 'bedrooms', 'bathrooms',
                    'square_feet', 'property_type', 'listing_url'
                ]
            ),
            api_key=api_key
        )
    
    async def aggregate_listings(
        self, 
        sources: List[Dict[str, str]], 
        filters: Dict = None
    ) -> List[Dict]:
        """
        Aggregate listings from multiple sources
        
        Args:
            sources: List of {'name': str, 'url': str, 'location': str}
            filters: Optional filters (price_min, price_max, bedrooms_min, etc.)
        
        Returns:
            List of normalized listings
        """
        
        all_listings = []
        
        for source in sources:
            logger.info(f"🏠 Crawling {source['name']} ({source['location']})...")
            
            try:
                # Execute crawl + scrape
                result = await self.workflow.execute(
                    start_urls=[source['url']]
                )
                
                # Add metadata
                for listing in result['data']:
                    listing['source'] = source['name']
                    listing['location'] = source['location']
                    listing['scraped_at'] = datetime.now().isoformat()
                
                all_listings.extend(result['data'])
                
                logger.info(
                    f"✅ {source['name']}: {len(result['data'])} listings "
                    f"from {result['crawl_metadata']['total_crawled']} pages"
                )
            
            except Exception as e:
                logger.error(f"❌ Failed to crawl {source['name']}: {e}")
                continue
        
        # Apply filters
        if filters:
            all_listings = self.filter_listings(all_listings, filters)
        
        # Deduplicate by address
        all_listings = self.deduplicate_by_address(all_listings)
        
        logger.info(f"📊 Total unique listings: {len(all_listings)}")
        
        return all_listings
    
    def filter_listings(self, listings: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to listings"""
        
        filtered = listings
        
        if filters.get('price_min'):
            filtered = [l for l in filtered if l.get('price', 0) >= filters['price_min']]
        
        if filters.get('price_max'):
            filtered = [l for l in filtered if l.get('price', float('inf')) <= filters['price_max']]
        
        if filters.get('bedrooms_min'):
            filtered = [l for l in filtered if l.get('bedrooms', 0) >= filters['bedrooms_min']]
        
        if filters.get('property_type'):
            filtered = [
                l for l in filtered 
                if l.get('property_type', '').lower() == filters['property_type'].lower()
            ]
        
        logger.info(f"🔍 Filtered: {len(listings)} → {len(filtered)} listings")
        
        return filtered
    
    def deduplicate_by_address(self, listings: List[Dict]) -> List[Dict]:
        """Remove duplicate listings by address"""
        
        seen_addresses = set()
        unique = []
        
        for listing in listings:
            address = listing.get('address', '').lower().strip()
            if address and address not in seen_addresses:
                seen_addresses.add(address)
                unique.append(listing)
        
        logger.info(f"🗑️  Deduplicated: {len(listings)} → {len(unique)} listings")
        
        return unique
    
    async def export_to_database(self, listings: List[Dict], db_connection):
        """Export listings to database"""
        
        # Example: PostgreSQL
        async with db_connection.transaction():
            for listing in listings:
                await db_connection.execute("""
                    INSERT INTO listings (
                        address, price, bedrooms, bathrooms, square_feet,
                        property_type, listing_url, source, location, scraped_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (address) DO UPDATE SET
                        price = EXCLUDED.price,
                        scraped_at = EXCLUDED.scraped_at
                """, 
                    listing['address'],
                    listing['price'],
                    listing['bedrooms'],
                    listing['bathrooms'],
                    listing.get('square_feet'),
                    listing.get('property_type'),
                    listing['listing_url'],
                    listing['source'],
                    listing['location'],
                    listing['scraped_at']
                )
        
        logger.info(f"💾 Exported {len(listings)} listings to database")
    
    async def close(self):
        """Clean up resources"""
        self.workflow.close()


# Example usage
async def main():
    # Real estate sources to aggregate
    sources = [
        {
            "name": "Zillow",
            "url": "https://www.zillow.com/san-francisco-ca/",
            "location": "San Francisco, CA"
        },
        {
            "name": "Redfin",
            "url": "https://www.redfin.com/city/17151/CA/San-Francisco",
            "location": "San Francisco, CA"
        },
        {
            "name": "Realtor.com",
            "url": "https://www.realtor.com/realestateandhomes-search/San-Francisco_CA",
            "location": "San Francisco, CA"
        }
    ]
    
    # Filters
    filters = {
        "price_min": 500000,
        "price_max": 2000000,
        "bedrooms_min": 2,
        "property_type": "Condo"
    }
    
    # Initialize aggregator
    aggregator = RealEstateAggregator(api_key="your-openai-key")
    
    try:
        # Aggregate listings
        listings = await aggregator.aggregate_listings(sources, filters)
        
        # Export to JSON
        import json
        with open('listings.json', 'w') as f:
            json.dump(listings, f, indent=2)
        
        # Export to database (if configured)
        # import asyncpg
        # db = await asyncpg.connect('postgresql://...')
        # await aggregator.export_to_database(listings, db)
        # await db.close()
        
        print(f"✅ Successfully aggregated {len(listings)} unique listings")
    
    finally:
        await aggregator.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Example 3: News Article Aggregator with Sentiment Analysis

**Use Case**: Monitor news sites and analyze sentiment.

```python
"""
News Aggregator with Sentiment Analysis
Monitors news sites and analyzes article sentiment
"""

import asyncio
from typing import List, Dict
from datetime import datetime, timedelta
from universal_scraper import UniversalScraper
from universal_scraper.core import SchemaDefinition, SchemaField


class NewsAggregator:
    """Aggregate and analyze news articles"""
    
    def __init__(self, api_key: str):
        # Define article schema
        self.schema = SchemaDefinition(
            name="news_article",
            version="1.0",
            fields=[
                SchemaField("title", "string", required=True),
                SchemaField("author", "string", required=False),
                SchemaField("publish_date", "string", required=False),
                SchemaField("content", "string", required=True),
                SchemaField("category", "string", required=False),
                SchemaField("tags", "list", required=False),
                SchemaField("url", "string", required=True)
            ]
        )
        
        self.scraper = UniversalScraper(
            api_key=api_key,
            schema=self.schema,
            fetch_mode="hybrid",
            enable_cache=True
        )
    
    async def aggregate_news(
        self, 
        sources: List[Dict[str, str]], 
        keywords: List[str] = None
    ) -> List[Dict]:
        """
        Aggregate news articles from multiple sources
        
        Args:
            sources: List of {'name': str, 'url': str}
            keywords: Optional keywords to filter by
        
        Returns:
            List of articles with sentiment analysis
        """
        
        all_articles = []
        
        for source in sources:
            logger.info(f"📰 Scraping {source['name']}...")
            
            try:
                result = await self.scraper.scrape(
                    url=source['url'],
                    fields=['title', 'author', 'publish_date', 'content', 'url']
                )
                
                for article in result['data']:
                    article['source'] = source['name']
                    article['scraped_at'] = datetime.now().isoformat()
                
                all_articles.extend(result['data'])
                
                logger.info(f"✅ {source['name']}: {len(result['data'])} articles")
            
            except Exception as e:
                logger.error(f"❌ Failed to scrape {source['name']}: {e}")
                continue
        
        # Filter by keywords
        if keywords:
            all_articles = self.filter_by_keywords(all_articles, keywords)
        
        # Analyze sentiment
        for article in all_articles:
            article['sentiment'] = await self.analyze_sentiment(article['content'])
        
        logger.info(f"📊 Total articles: {len(all_articles)}")
        
        return all_articles
    
    def filter_by_keywords(self, articles: List[Dict], keywords: List[str]) -> List[Dict]:
        """Filter articles by keywords"""
        
        filtered = []
        
        for article in articles:
            content = (
                article.get('title', '') + ' ' + 
                article.get('content', '')
            ).lower()
            
            if any(keyword.lower() in content for keyword in keywords):
                filtered.append(article)
        
        logger.info(f"🔍 Filtered by keywords: {len(articles)} → {len(filtered)}")
        
        return filtered
    
    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using LLM"""
        
        from litellm import acompletion
        
        prompt = f"""Analyze the sentiment of this text and return a JSON response:

Text: {text[:500]}...

Return JSON with:
- sentiment: "positive", "negative", or "neutral"
- score: float between -1.0 (very negative) and 1.0 (very positive)
- reasoning: brief explanation

JSON:"""
        
        try:
            response = await acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "unknown", "score": 0.0, "reasoning": "Analysis failed"}
    
    async def generate_summary(self, articles: List[Dict]) -> Dict:
        """Generate summary of aggregated news"""
        
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        total_score = 0
        
        for article in articles:
            sentiment = article.get('sentiment', {})
            sentiment_counts[sentiment.get('sentiment', 'neutral')] += 1
            total_score += sentiment.get('score', 0)
        
        avg_score = total_score / len(articles) if articles else 0
        
        return {
            "total_articles": len(articles),
            "sentiment_distribution": sentiment_counts,
            "average_sentiment_score": avg_score,
            "sources": len(set(a['source'] for a in articles)),
            "date_range": self._get_date_range(articles)
        }
    
    def _get_date_range(self, articles: List[Dict]) -> Dict:
        """Get date range of articles"""
        
        dates = [
            datetime.fromisoformat(a['publish_date'])
            for a in articles
            if a.get('publish_date')
        ]
        
        if dates:
            return {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat()
            }
        
        return {"earliest": None, "latest": None}
    
    async def close(self):
        """Clean up resources"""
        await self.scraper.close()


# Example usage
async def main():
    # News sources
    sources = [
        {"name": "TechCrunch", "url": "https://techcrunch.com/"},
        {"name": "The Verge", "url": "https://www.theverge.com/tech"},
        {"name": "Ars Technica", "url": "https://arstechnica.com/"}
    ]
    
    # Keywords to filter
    keywords = ["AI", "artificial intelligence", "machine learning", "ChatGPT"]
    
    # Initialize aggregator
    aggregator = NewsAggregator(api_key="your-openai-key")
    
    try:
        # Aggregate news
        articles = await aggregator.aggregate_news(sources, keywords)
        
        # Generate summary
        summary = await aggregator.generate_summary(articles)
        
        print("\n📊 News Summary:")
        print(f"Total Articles: {summary['total_articles']}")
        print(f"Sentiment Distribution: {summary['sentiment_distribution']}")
        print(f"Average Sentiment: {summary['average_sentiment_score']:.2f}")
        
        # Display top positive and negative articles
        positive = sorted(
            [a for a in articles if a.get('sentiment', {}).get('sentiment') == 'positive'],
            key=lambda x: x.get('sentiment', {}).get('score', 0),
            reverse=True
        )[:3]
        
        negative = sorted(
            [a for a in articles if a.get('sentiment', {}).get('sentiment') == 'negative'],
            key=lambda x: x.get('sentiment', {}).get('score', 0)
        )[:3]
        
        print("\n🟢 Top Positive Articles:")
        for article in positive:
            print(f"- {article['title']}")
            print(f"  Score: {article['sentiment']['score']:.2f}")
            print(f"  {article['sentiment']['reasoning']}")
        
        print("\n🔴 Top Negative Articles:")
        for article in negative:
            print(f"- {article['title']}")
            print(f"  Score: {article['sentiment']['score']:.2f}")
            print(f"  {article['sentiment']['reasoning']}")
        
        # Export to JSON
        import json
        with open('news_articles.json', 'w') as f:
            json.dump(articles, f, indent=2)
    
    finally:
        await aggregator.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Example 4: Job Listings Aggregator

**Use Case**: Aggregate job listings from multiple job boards.

```python
"""
Job Listings Aggregator
Aggregates and normalizes job listings from multiple sources
"""

import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from universal_scraper.orchestrator import UniversalWorkflow, WorkflowConfig, WorkflowMode
from universal_scraper.crawler import CrawlConfig
from universal_scraper.core import SchemaDefinition, SchemaField


class JobAggregator:
    """Aggregate job listings from multiple job boards"""
    
    def __init__(self, api_key: str):
        # Define job listing schema
        self.schema = SchemaDefinition(
            name="job_listing",
            version="1.0",
            fields=[
                SchemaField("job_title", "string", required=True),
                SchemaField("company_name", "string", required=True),
                SchemaField("location", "string", required=True),
                SchemaField("job_type", "string", required=False),  # Full-time, Part-time, Contract
                SchemaField("remote_option", "string", required=False),  # Remote, Hybrid, On-site
                SchemaField("salary_min", "float", required=False),
                SchemaField("salary_max", "float", required=False),
                SchemaField("salary_currency", "string", required=False),
                SchemaField("description", "string", required=False),
                SchemaField("requirements", "list", required=False),
                SchemaField("benefits", "list", required=False),
                SchemaField("apply_url", "string", required=True),
                SchemaField("posted_date", "string", required=False)
            ]
        )
        
        self.workflow = UniversalWorkflow(
            config=WorkflowConfig(
                mode=WorkflowMode.CRAWL_THEN_SCRAPE,
                crawl_config=CrawlConfig(
                    mode='smart',
                    max_depth=2,
                    max_pages=200,
                    handle_pagination=True,
                    url_patterns=[
                        r'/jobs/\d+',
                        r'/job/[\w-]+',
                        r'/careers/[\w-]+'
                    ]
                ),
                schema=self.schema,
                fields=[
                    'job_title', 'company_name', 'location', 'salary_min',
                    'salary_max', 'job_type', 'remote_option', 'apply_url'
                ]
            ),
            api_key=api_key
        )
    
    async def aggregate_jobs(
        self,
        sources: List[Dict[str, str]],
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Aggregate job listings from multiple sources
        
        Args:
            sources: List of job board URLs
            filters: Optional filters (keywords, location, salary, etc.)
        """
        
        all_jobs = []
        
        for source in sources:
            logger.info(f"💼 Crawling {source['name']}...")
            
            try:
                result = await self.workflow.execute(
                    start_urls=[source['url']]
                )
                
                # Add metadata
                for job in result['data']:
                    job['source'] = source['name']
                    job['scraped_at'] = datetime.now().isoformat()
                
                all_jobs.extend(result['data'])
                
                logger.info(
                    f"✅ {source['name']}: {len(result['data'])} jobs "
                    f"from {result['crawl_metadata']['total_crawled']} pages"
                )
            
            except Exception as e:
                logger.error(f"❌ Failed to crawl {source['name']}: {e}")
                continue
        
        # Apply filters
        if filters:
            all_jobs = self.filter_jobs(all_jobs, filters)
        
        # Deduplicate
        all_jobs = self.deduplicate_jobs(all_jobs)
        
        # Enrich with ML scores
        all_jobs = await self.score_job_matches(all_jobs, filters)
        
        logger.info(f"📊 Total unique jobs: {len(all_jobs)}")
        
        return all_jobs
    
    def filter_jobs(self, jobs: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to job listings"""
        
        filtered = jobs
        
        # Keyword filter
        if filters.get('keywords'):
            keywords = [k.lower() for k in filters['keywords']]
            filtered = [
                j for j in filtered
                if any(
                    keyword in j.get('job_title', '').lower() or
                    keyword in j.get('description', '').lower()
                    for keyword in keywords
                )
            ]
        
        # Location filter
        if filters.get('location'):
            location = filters['location'].lower()
            filtered = [
                j for j in filtered
                if location in j.get('location', '').lower()
            ]
        
        # Remote filter
        if filters.get('remote_only'):
            filtered = [
                j for j in filtered
                if 'remote' in j.get('remote_option', '').lower()
            ]
        
        # Salary filter
        if filters.get('salary_min'):
            filtered = [
                j for j in filtered
                if j.get('salary_min', 0) >= filters['salary_min']
            ]
        
        logger.info(f"🔍 Filtered: {len(jobs)} → {len(filtered)} jobs")
        
        return filtered
    
    def deduplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate job listings"""
        
        seen = set()
        unique = []
        
        for job in jobs:
            # Create unique key from title + company + location
            key = (
                job.get('job_title', '').lower(),
                job.get('company_name', '').lower(),
                job.get('location', '').lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(job)
        
        logger.info(f"🗑️  Deduplicated: {len(jobs)} → {len(unique)} jobs")
        
        return unique
    
    async def score_job_matches(
        self, 
        jobs: List[Dict], 
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Score how well each job matches user preferences using LLM"""
        
        if not filters or not filters.get('user_profile'):
            return jobs
        
        from litellm import acompletion
        
        user_profile = filters['user_profile']
        
        for job in jobs:
            prompt = f"""Score how well this job matches the candidate profile (0-100):

Candidate Profile:
{user_profile}

Job:
- Title: {job.get('job_title')}
- Company: {job.get('company_name')}
- Location: {job.get('location')}
- Remote: {job.get('remote_option')}
- Salary: {job.get('salary_min')}-{job.get('salary_max')}
- Type: {job.get('job_type')}

Return JSON: {{"score": 0-100, "reasoning": "brief explanation"}}
"""
            
            try:
                response = await acompletion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                import json
                result = json.loads(response.choices[0].message.content)
                job['match_score'] = result['score']
                job['match_reasoning'] = result['reasoning']
            
            except Exception as e:
                logger.warning(f"Failed to score job {job['job_title']}: {e}")
                job['match_score'] = 0
        
        # Sort by match score
        jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return jobs
    
    async def close(self):
        """Clean up resources"""
        self.workflow.close()


# Example usage
async def main():
    # Job board sources
    sources = [
        {"name": "LinkedIn", "url": "https://www.linkedin.com/jobs/search/?keywords=software%20engineer"},
        {"name": "Indeed", "url": "https://www.indeed.com/jobs?q=software+engineer"},
        {"name": "Glassdoor", "url": "https://www.glassdoor.com/Job/software-engineer-jobs-SRCH_KO0,17.htm"}
    ]
    
    # Filters
    filters = {
        "keywords": ["python", "machine learning", "AI"],
        "location": "san francisco",
        "remote_only": True,
        "salary_min": 120000,
        "user_profile": """
        Software Engineer with 5 years experience.
        Skills: Python, Machine Learning, PyTorch, AWS
        Looking for: Senior ML Engineer role, remote, $150k+
        Interests: AI safety, computer vision, NLP
        """
    }
    
    # Initialize aggregator
    aggregator = JobAggregator(api_key="your-openai-key")
    
    try:
        # Aggregate jobs
        jobs = await aggregator.aggregate_jobs(sources, filters)
        
        # Display top matches
        print("\n🎯 Top 10 Job Matches:\n")
        
        for i, job in enumerate(jobs[:10], 1):
            print(f"{i}. {job['job_title']} at {job['company_name']}")
            print(f"   📍 {job['location']} | {job.get('remote_option', 'N/A')}")
            
            if job.get('salary_min'):
                print(f"   💰 ${job['salary_min']:,.0f} - ${job.get('salary_max', 0):,.0f}")
            
            print(f"   ⭐ Match Score: {job.get('match_score', 0)}/100")
            print(f"   💭 {job.get('match_reasoning', 'N/A')}")
            print(f"   🔗 {job['apply_url']}\n")
        
        # Export to JSON
        import json
        with open('job_listings.json', 'w') as f:
            json.dump(jobs, f, indent=2)
        
        print(f"✅ Saved {len(jobs)} jobs to job_listings.json")
    
    finally:
        await aggregator.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

These examples demonstrate:

1. **Production-ready code**: Error handling, logging, resource cleanup
2. **Schema management**: Consistent data structures across sources
3. **Filtering & deduplication**: Data quality assurance
4. **AI integration**: Sentiment analysis, job matching
5. **Multi-source aggregation**: Unified data from diverse sources
6. **Export capabilities**: JSON, databases, APIs

All examples are ready to deploy in production with minimal modifications.








