"""
Hybrid Markdown Extractor - Combines structured data extraction with markdown conversion

This addresses the 8 critical edge cases where pure markdown conversion loses data:
1. Data Attributes - Extracted before markdown conversion
2. JSON-LD Scripts - Extracted in Stage 1 (highest priority)
3. Form/Select Data - Extracted before forms are removed
4. Complex Tables - Converted to structured JSON
5. Meta Tags - Extracted for additional context
6. Hidden Inputs - Captured for product IDs, pricing, etc.

Based on analysis of ScrapeGraphAI and Oxylabs approaches.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Comment
import html2text

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Container for extracted content from hybrid extraction"""
    markdown: str
    json_ld: List[Dict] = field(default_factory=list)
    data_attributes: Dict[str, List[str]] = field(default_factory=dict)
    form_data: Dict[str, Any] = field(default_factory=dict)
    tables: List[List[List[str]]] = field(default_factory=list)
    meta_tags: Dict[str, str] = field(default_factory=dict)
    hidden_inputs: Dict[str, str] = field(default_factory=dict)
    css_class_data: Dict[str, List[str]] = field(default_factory=dict)  # CSS class-based data
    labeled_numbers: Dict[str, List[str]] = field(default_factory=dict)  # NEW: "5 answers" -> answers: [5]
    script_data: List[str] = field(default_factory=list)  # NEW: window/document variable data (ScrapeGraphAI style)
    button_data: Dict[str, List[str]] = field(default_factory=dict)  # NEW: Upvote/action button values
    source_url: str = ""
    
    @property
    def has_structured_data(self) -> bool:
        """Check if we have any structured data"""
        return bool(self.json_ld or self.data_attributes or self.form_data or self.hidden_inputs or self.css_class_data or self.labeled_numbers or self.script_data or self.button_data)
    
    def get_metadata_summary(self) -> str:
        """Get a summary of extracted metadata for LLM context"""
        parts = []
        
        if self.data_attributes:
            attrs = []
            for key, values in self.data_attributes.items():
                # Only include unique values
                unique_vals = list(set(values))[:5]  # Limit to 5 per key
                attrs.append(f"{key}: {', '.join(unique_vals)}")
            if attrs:
                parts.append(f"DATA ATTRIBUTES:\n" + "\n".join(attrs))
        
        if self.form_data:
            forms = []
            for name, values in self.form_data.items():
                if isinstance(values, list):
                    forms.append(f"{name}: {', '.join(str(v) for v in values[:10])}")
                else:
                    forms.append(f"{name}: {values}")
            if forms:
                parts.append(f"FORM OPTIONS:\n" + "\n".join(forms))
        
        if self.hidden_inputs:
            inputs = [f"{k}: {v}" for k, v in list(self.hidden_inputs.items())[:10]]
            if inputs:
                parts.append(f"HIDDEN VALUES:\n" + "\n".join(inputs))
        
        if self.meta_tags:
            # Only include relevant meta tags
            relevant_keys = ['title', 'description', 'price', 'product', 'og:title', 
                           'og:description', 'og:price', 'product:price:amount']
            metas = [f"{k}: {v}" for k, v in self.meta_tags.items() 
                    if any(rk in k.lower() for rk in relevant_keys)][:10]
            if metas:
                parts.append(f"META TAGS:\n" + "\n".join(metas))
        
        # Include CSS class-based data (ratings, scores, votes)
        # Make ratings more explicit for LLM understanding
        if self.css_class_data:
            css_items = []
            for key, values in self.css_class_data.items():
                unique_vals = list(set(values))[:10]
                
                # Convert word ratings to numbers for LLM understanding
                if 'rating' in key.lower() or 'star' in key.lower():
                    word_to_num = {
                        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                        'zero': '0', 'half': '0.5'
                    }
                    converted = []
                    for v in unique_vals:
                        v_lower = v.lower()
                        if v_lower in word_to_num:
                            converted.append(f"{v}={word_to_num[v_lower]} stars")
                        else:
                            converted.append(v)
                    css_items.append(f"{key}: {', '.join(converted)}")
                    css_items.append(f"  ↳ NOTE: Use numeric values (1-5) for rating field")
                else:
                    css_items.append(f"{key}: {', '.join(unique_vals)}")
            if css_items:
                parts.append(f"CSS CLASS DATA (ratings, votes, scores):\n" + "\n".join(css_items))
        
        # Include labeled numbers (e.g., "5 answers", "120 votes")
        if self.labeled_numbers:
            labeled_items = []
            for key, values in self.labeled_numbers.items():
                unique_vals = list(set(values))[:10]
                labeled_items.append(f"{key}: {', '.join(unique_vals)}")
            if labeled_items:
                parts.append(f"LABELED NUMBERS (votes, answers, views, etc.):\n" + "\n".join(labeled_items))
        
        # Include script data (window/document variables) - ScrapeGraphAI style
        if self.script_data:
            script_items = self.script_data[:5]  # Limit to 5 items
            if script_items:
                parts.append(f"SCRIPT DATA (JavaScript variables):\n" + "\n".join(script_items))
        
        # Include button/action data (upvotes, likes, etc.) - captured before buttons stripped
        if self.button_data:
            button_items = []
            for action, values in self.button_data.items():
                unique_vals = list(set(values))[:15]  # Up to 15 unique values
                button_items.append(f"{action}: {', '.join(unique_vals)}")
            if button_items:
                parts.append(f"BUTTON/ACTION DATA (upvotes, likes, votes):\n" + "\n".join(button_items))
        
        return "\n\n".join(parts) if parts else ""


class HybridMarkdownExtractor:
    """
    Hybrid extraction pipeline that captures structured data before markdown conversion.
    
    Pipeline:
    1. Extract JSON-LD (highest priority - often contains complete data)
    2. Extract data-* attributes (hidden values like prices, stock, ratings)
    3. Extract form/select data (product variants, sizes, colors)
    4. Extract tables as structured data
    5. Extract meta tags
    6. Convert remaining HTML to markdown
    7. Return combined result for LLM processing
    """
    
    # Tags to remove completely after extraction
    NOISE_TAGS = [
        'script',    # JavaScript (except JSON-LD, extracted first)
        'style',     # CSS styles
        'noscript',  # Noscript fallbacks
        'iframe',    # Embedded frames
        'embed',     # Embedded objects
        'object',    # Object embeds
        'nav',       # Navigation menus
        'header',    # Page headers
        'footer',    # Page footers
        'aside',     # Sidebar content
        'svg',       # Icons and graphics
    ]
    
    # Noise patterns in class/ID
    NOISE_PATTERNS = [
        'advertisement', 'ad-container', 'ad-banner', 'google-ad',
        'sponsored', 'cookie-consent', 'gdpr-notice',
        'social-share', 'share-button', 'social-links',
        'newsletter', 'email-signup', 'subscribe',
        'related-posts', 'related-content', 'sidebar-widget',
        'breadcrumb', 'pagination', 'mobile-menu',
        'author-bio', 'comment-form', 'comments-section',
    ]
    
    # CSS class patterns that contain valuable data (ratings, votes, scores)
    # Pattern: (class_pattern, extraction_type, value_extractor)
    CSS_DATA_PATTERNS = [
        # Star ratings - look for "star-rating One/Two/Three/Four/Five"
        (r'star[-_]?rating\s+(\w+)', 'rating', lambda m: m.group(1).lower()),
        # Numeric ratings in class - "rating-3", "stars-4"
        (r'(?:rating|stars?)[-_]?(\d+)', 'rating', lambda m: m.group(1)),
        # Vote/score classes - "score-100", "votes-50"
        (r'(?:score|votes?|points?)[-_]?(\d+)', 'score', lambda m: m.group(1)),
        # Active/selected state for ratings
        (r'(?:active|selected|filled)[-_]?(\d+)', 'rating_active', lambda m: m.group(1)),
    ]
    
    # Text-to-number mappings for word ratings
    WORD_TO_NUMBER = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'zero': '0', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    
    def __init__(self, ignore_links: bool = False, ignore_images: bool = True):
        """
        Initialize the hybrid extractor.
        
        Args:
            ignore_links: Whether to ignore links in markdown output
            ignore_images: Whether to ignore images in markdown output
        """
        self.h = html2text.HTML2Text()
        self.h.unicode_snob = True  # Preserve unicode characters
        self.h.ignore_links = ignore_links
        self.h.ignore_images = ignore_images
        self.h.body_width = 0  # Don't wrap lines
        self.h.ignore_emphasis = True  # Don't add markdown emphasis
        self.h.single_line_break = True  # Use single line breaks
    
    def extract(self, html: str, url: str = "") -> ExtractedContent:
        """
        Extract content using hybrid approach.
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            ExtractedContent with markdown + structured data
        """
        original_size = len(html)
        logger.info(f" Hybrid extraction starting ({original_size:,} bytes)")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Stage 1: Extract JSON-LD (BEFORE any cleaning)
        json_ld = self._extract_json_ld(soup)
        if json_ld:
            logger.info(f"    Extracted {len(json_ld)} JSON-LD objects")
        
        # Stage 1.5: Extract script data (ScrapeGraphAI style - window/document variables)
        script_data = self._extract_script_data(soup)
        if script_data:
            logger.info(f"    Extracted {len(script_data)} script data items")
        
        # Stage 2: Extract structured data (BEFORE removing forms/buttons)
        data_attrs = self._extract_data_attributes(soup)
        form_data = self._extract_form_data(soup)
        hidden_inputs = self._extract_hidden_inputs(soup)
        tables = self._extract_tables(soup)
        meta_tags = self._extract_meta_tags(soup)
        css_class_data = self._extract_css_class_data(soup)  # CSS class-based data
        labeled_numbers = self._extract_labeled_numbers(soup)  # Universal labeled number extraction
        button_data = self._extract_button_data(soup)  # Button/vote data (BEFORE buttons removed)
        
        if data_attrs:
            logger.info(f"    Extracted {len(data_attrs)} data attribute types")
        if form_data:
            logger.info(f"    Extracted {len(form_data)} form fields")
        if hidden_inputs:
            logger.info(f"    Extracted {len(hidden_inputs)} hidden inputs")
        if tables:
            logger.info(f"    Extracted {len(tables)} tables")
        if css_class_data:
            logger.info(f"    Extracted {len(css_class_data)} CSS class data types (ratings/votes/scores)")
        if labeled_numbers:
            logger.info(f"    Extracted {len(labeled_numbers)} labeled number types (votes/answers/views)")
        if button_data:
            logger.info(f"    Extracted {len(button_data)} button/action data types (upvotes/votes)")
        
        # Stage 2.5: Preserve vote button content BEFORE removing buttons
        # This injects the vote count as visible text so it appears in markdown
        self._preserve_vote_buttons(soup)
        
        # Stage 3: Clean HTML (now safe to remove forms, scripts, etc.)
        self._remove_noise_tags(soup)
        self._remove_noise_elements(soup)
        self._remove_comments(soup)
        
        # Stage 4: Convert to markdown
        cleaned_html = str(soup)
        markdown = self.h.handle(cleaned_html)
        markdown = self._clean_markdown(markdown)
        
        logger.info(f"    Converted to markdown ({len(markdown):,} chars)")
        
        return ExtractedContent(
            markdown=markdown,
            json_ld=json_ld,
            data_attributes=data_attrs,
            form_data=form_data,
            tables=tables,
            meta_tags=meta_tags,
            hidden_inputs=hidden_inputs,
            css_class_data=css_class_data,
            labeled_numbers=labeled_numbers,
            script_data=script_data,  # ScrapeGraphAI style
            button_data=button_data,  # Button/vote data (captured before cleaning)
            source_url=url
        )
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all JSON-LD structured data"""
        json_ld_list = []
        
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                if script.string:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        json_ld_list.extend(data)
                    else:
                        json_ld_list.append(data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"   Failed to parse JSON-LD: {e}")
        
        return json_ld_list
    
    def _extract_script_data(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract data from script tags (ScrapeGraphAI approach).
        
        Extracts:
        - JSON data from variable assignments (const/let/var x = {...})
        - window.* and document.* variable assignments
        - Useful inline data that might contain product info, prices, etc.
        """
        script_content = []
        
        for script in soup.find_all("script"):
            # Skip JSON-LD (already extracted separately)
            if script.get('type') == 'application/ld+json':
                continue
                
            content = script.string
            if not content:
                continue
            
            try:
                # Pattern 1: Extract JSON objects from variable assignments
                # Matches: const data = {...}, let config = {...}, var settings = {...}
                json_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*(\{[\s\S]*?\});?"
                json_matches = re.findall(json_pattern, content)
                
                for var_name, potential_json in json_matches:
                    try:
                        # Try to parse as JSON
                        parsed = json.loads(potential_json)
                        if parsed and isinstance(parsed, dict) and len(parsed) > 0:
                            # Skip large objects (likely framework data)
                            json_str = json.dumps(parsed, indent=2)
                            if len(json_str) < 2000:
                                script_content.append(f"Variable '{var_name}': {json_str}")
                    except json.JSONDecodeError:
                        pass
                
                # Pattern 2: Extract window.* and document.* assignments
                # ScrapeGraphAI specifically looks for these
                if "window." in content or "document." in content:
                    data_pattern = r"(?:window|document)\.(\w+)\s*=\s*([^;]{1,500});"
                    data_matches = re.findall(data_pattern, content)
                    
                    for var_name, var_value in data_matches:
                        # Skip function definitions
                        if 'function' in var_value or '=>' in var_value:
                            continue
                        # Skip common non-data properties
                        if var_name in ['addEventListener', 'onload', 'onclick', 'dataLayer']:
                            continue
                        
                        value = var_value.strip()
                        if value and len(value) < 500:
                            script_content.append(f"{var_name}: {value}")
                
                # Pattern 3: Look for __INITIAL_STATE__ or similar (common in React/Vue apps)
                state_patterns = [
                    r'__INITIAL_STATE__\s*=\s*(\{[\s\S]*?\});',
                    r'__NEXT_DATA__\s*=\s*(\{[\s\S]*?\});',
                    r'__PRELOADED_STATE__\s*=\s*(\{[\s\S]*?\});',
                    r'window\.__data\s*=\s*(\{[\s\S]*?\});',
                ]
                
                for pattern in state_patterns:
                    state_match = re.search(pattern, content)
                    if state_match:
                        try:
                            state_data = json.loads(state_match.group(1))
                            # Only include if it has useful-looking data
                            if state_data and isinstance(state_data, dict):
                                json_str = json.dumps(state_data, indent=2)
                                if len(json_str) < 5000:  # Allow larger state objects
                                    script_content.append(f"Initial State: {json_str[:2000]}...")
                        except json.JSONDecodeError:
                            pass
                            
            except Exception as e:
                # If script is small, include it as raw content (ScrapeGraphAI does this)
                if len(content) < 500:
                    script_content.append(f"Script content: {content.strip()[:400]}")
        
        return script_content[:10]  # Limit to 10 items to avoid overwhelming the LLM
    
    def _extract_data_attributes(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all data-* attributes from elements"""
        data_attrs: Dict[str, List[str]] = {}
        
        for elem in soup.find_all(attrs=True):
            for attr, value in elem.attrs.items():
                if attr.startswith('data-') and value:
                    # Clean up the key
                    key = attr.replace('data-', '').replace('-', '_')
                    
                    # Convert value to string if needed
                    if isinstance(value, list):
                        value = ' '.join(value)
                    else:
                        value = str(value)
                    
                    # Skip empty or very long values
                    if not value or len(value) > 500:
                        continue
                    
                    # Skip common non-data attributes
                    if key in ['testid', 'test', 'ga', 'analytics', 'track', 'event']:
                        continue
                    
                    if key not in data_attrs:
                        data_attrs[key] = []
                    
                    if value not in data_attrs[key]:
                        data_attrs[key].append(value)
        
        return data_attrs
    
    def _extract_form_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract form field data (especially select/dropdown options)"""
        form_data: Dict[str, Any] = {}
        
        # Extract select dropdowns (sizes, colors, variants)
        for select in soup.find_all('select'):
            name = select.get('name') or select.get('id') or 'unnamed_select'
            options = []
            
            for option in select.find_all('option'):
                opt_value = option.get('value', '')
                opt_text = option.get_text(strip=True)
                
                if opt_text and opt_text.lower() not in ['select', 'choose', '--', '-']:
                    options.append({
                        'value': opt_value,
                        'text': opt_text,
                        'selected': option.has_attr('selected')
                    })
            
            if options:
                form_data[name] = options
        
        # Extract radio button options
        radio_groups: Dict[str, List[Dict]] = {}
        for radio in soup.find_all('input', type='radio'):
            name = radio.get('name', 'unnamed_radio')
            value = radio.get('value', '')
            label = self._find_label_for_input(soup, radio)
            
            if name not in radio_groups:
                radio_groups[name] = []
            
            radio_groups[name].append({
                'value': value,
                'label': label,
                'checked': radio.has_attr('checked')
            })
        
        form_data.update(radio_groups)
        
        return form_data
    
    def _extract_hidden_inputs(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract hidden input values (often contain product IDs, prices, etc.)"""
        hidden = {}
        
        for input_elem in soup.find_all('input', type='hidden'):
            name = input_elem.get('name') or input_elem.get('id')
            value = input_elem.get('value', '')
            
            if name and value and len(value) < 200:
                # Skip common non-useful hidden fields
                skip_patterns = ['csrf', 'token', 'nonce', 'session', 'captcha']
                if not any(pattern in name.lower() for pattern in skip_patterns):
                    hidden[name] = value
        
        return hidden
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables as structured data"""
        tables = []
        
        for table in soup.find_all('table'):
            rows = []
            
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text(strip=True)
                    cells.append(cell_text)
                
                if cells and any(c for c in cells):  # Skip empty rows
                    rows.append(cells)
            
            if rows and len(rows) > 1:  # Only include tables with actual data
                tables.append(rows)
        
        return tables
    
    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract relevant meta tags"""
        meta = {}
        
        for tag in soup.find_all('meta'):
            name = tag.get('name') or tag.get('property') or tag.get('itemprop')
            content = tag.get('content')
            
            if name and content and len(content) < 500:
                meta[name] = content
        
        # Also extract title
        title_tag = soup.find('title')
        if title_tag:
            meta['title'] = title_tag.get_text(strip=True)
        
        return meta
    
    def _extract_css_class_data(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Extract data encoded in CSS classes (ratings, votes, scores).
        
        Many sites encode data in class names:
        - Books to Scrape: <p class="star-rating Three">
        - E-commerce: <span class="rating-4">
        - News sites: <span class="score-123">
        """
        css_data: Dict[str, List[str]] = {}
        
        for elem in soup.find_all(attrs={'class': True}):
            classes = elem.get('class', [])
            if not classes:
                continue
            
            # Join all classes for pattern matching
            class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
            
            # Check each CSS data pattern
            for pattern, data_type, value_extractor in self.CSS_DATA_PATTERNS:
                match = re.search(pattern, class_str, re.IGNORECASE)
                if match:
                    try:
                        value = value_extractor(match)
                        
                        # Convert word ratings to numbers
                        if value.lower() in self.WORD_TO_NUMBER:
                            value = self.WORD_TO_NUMBER[value.lower()]
                        
                        if data_type not in css_data:
                            css_data[data_type] = []
                        
                        # Also try to get associated text for context
                        parent_text = elem.get_text(strip=True)[:100] if elem.get_text(strip=True) else ""
                        
                        entry = f"{value}"
                        if parent_text and parent_text != value:
                            entry = f"{value} (near: {parent_text})"
                        
                        if entry not in css_data[data_type]:
                            css_data[data_type].append(entry)
                    except Exception:
                        continue
        
        return css_data
    
    def _extract_labeled_numbers(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Extract labeled numbers from HTML text.
        
        Universal patterns found across many sites:
        - Stack Overflow: "5 answers", "0 votes", "1k views"  
        - Hacker News: "42 points", "15 comments"
        - Product Hunt: "120 upvotes"
        - E-commerce: "5 reviews", "3 in stock"
        
        This extracts number+label pairs from the visible text.
        """
        labeled_numbers: Dict[str, List[str]] = {}
        
        # Common label patterns (singular and plural forms)
        # Pattern format: regex to find "NUMBER LABEL" or "LABEL: NUMBER"
        LABEL_PATTERNS = [
            # Votes/Answers/Views (Stack Overflow, forums)
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(votes?|vote)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(answers?|answer)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(views?|view)',
            # Social engagement
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(upvotes?|upvote)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(downvotes?|downvote)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(likes?|like)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(comments?|comment)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(points?|point)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(shares?|share)',
            # Reviews/Ratings
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(reviews?|review)',
            r'(\d+(?:[\.,]\d+)?)\s*(?:/\s*5)?\s*(?:stars?|)',
            # Stock/Availability  
            r'(\d+)\s*(?:in\s+stock|available)',
            # Followers/Following
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(followers?|follower)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(following)',
            # Repository stats (GitHub-like)
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(stars?|)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(forks?|fork)',
            r'(\d+(?:[\.,]\d+)?[kKmM]?)\s*(watchers?|watcher)',
            # Reverse patterns: "Label: NUMBER"
            r'votes?[:\s]+(\d+(?:[\.,]\d+)?[kKmM]?)',
            r'answers?[:\s]+(\d+(?:[\.,]\d+)?[kKmM]?)',
            r'views?[:\s]+(\d+(?:[\.,]\d+)?[kKmM]?)',
        ]
        
        # Get all text content from the page
        text_elements = soup.find_all(text=True)
        
        for text_elem in text_elements:
            text = text_elem.strip()
            if not text or len(text) > 200:  # Skip empty or very long text
                continue
            
            for pattern in LABEL_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # Pattern with number first (e.g., "5 answers")
                        if len(match) >= 2:
                            number = match[0]
                            label = match[1].lower().rstrip('s')  # Normalize: answers -> answer
                        else:
                            number = match[0]
                            label = "count"
                    else:
                        number = match
                        # Extract label from pattern
                        label_match = re.search(r'(votes?|answers?|views?|comments?|points?|reviews?|likes?|shares?|stars?|forks?)', pattern, re.IGNORECASE)
                        label = label_match.group(1).rstrip('s') if label_match else "count"
                    
                    # Normalize label
                    label = label.lower().rstrip('s')
                    
                    if label not in labeled_numbers:
                        labeled_numbers[label] = []
                    
                    if number not in labeled_numbers[label]:
                        labeled_numbers[label].append(number)
        
        # Also check for stat-like elements with specific class patterns
        # (common in Stack Overflow, GitHub, etc.)
        stat_selectors = [
            ('stats', 'item'),
            ('stat', 'value'),
            ('counter', 'value'),
            ('metric', 'value'),
        ]
        
        for container_class, value_class in stat_selectors:
            for container in soup.find_all(class_=re.compile(container_class, re.IGNORECASE)):
                # Look for number + label pairs within this container
                text = container.get_text(separator=' ', strip=True)
                for pattern in LABEL_PATTERNS[:8]:  # Check main patterns
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            number, label = match[0], match[1].lower().rstrip('s')
                            if label not in labeled_numbers:
                                labeled_numbers[label] = []
                            if number not in labeled_numbers[label]:
                                labeled_numbers[label].append(number)
        
        return labeled_numbers
    
    def _extract_button_data(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Extract data from buttons/interactive elements BEFORE they get stripped.
        
        Critical for sites like Product Hunt where upvotes are in button elements:
        - Button text: "Upvote (290)" or just "290"
        - Button with vote-related classes: "upvote", "vote", "like"
        - Aria-labels with numbers: aria-label="290 votes"
        
        This runs BEFORE noise cleaning removes <button> elements.
        """
        button_data: Dict[str, List[str]] = {}
        
        # 1. Find buttons with vote/upvote/like in class name
        vote_buttons = soup.find_all('button', class_=re.compile(r'vote|upvote|like|reaction', re.IGNORECASE))
        for btn in vote_buttons:
            text = btn.get_text(strip=True)
            # Extract number from "Upvote (290)" or just "290"
            number_match = re.search(r'(\d+(?:[\.,]\d+)?[kKmM]?)', text)
            if number_match:
                number = number_match.group(1)
                if 'upvote' not in button_data:
                    button_data['upvote'] = []
                if number not in button_data['upvote']:
                    button_data['upvote'].append(number)
        
        # 2. Find buttons that contain only/mostly a number (vote counts)
        all_buttons = soup.find_all('button')
        for btn in all_buttons:
            text = btn.get_text(strip=True)
            # Button with just a number (1-4 digits, possibly with k/M suffix)
            if re.match(r'^\d{1,4}[kKmM]?$', text):
                if 'vote_count' not in button_data:
                    button_data['vote_count'] = []
                if text not in button_data['vote_count']:
                    button_data['vote_count'].append(text)
            # Button with "Upvote (N)" pattern
            elif 'upvote' in text.lower():
                number_match = re.search(r'\((\d+)\)', text)
                if number_match:
                    number = number_match.group(1)
                    if 'upvote' not in button_data:
                        button_data['upvote'] = []
                    if number not in button_data['upvote']:
                        button_data['upvote'].append(number)
        
        # 3. Check aria-labels for vote information
        vote_aria = soup.find_all(attrs={'aria-label': re.compile(r'\d+.*(?:vote|upvote|like)', re.IGNORECASE)})
        for elem in vote_aria:
            label = elem.get('aria-label', '')
            number_match = re.search(r'(\d+)', label)
            if number_match:
                number = number_match.group(1)
                if 'upvote' not in button_data:
                    button_data['upvote'] = []
                if number not in button_data['upvote']:
                    button_data['upvote'].append(number)
        
        # 4. Check data-* attributes for vote counts
        for attr in ['data-vote', 'data-votes', 'data-count', 'data-upvotes', 'data-score']:
            elements = soup.find_all(attrs={attr: True})
            for elem in elements:
                value = elem.get(attr)
                if value and re.match(r'^\d+', str(value)):
                    action = attr.replace('data-', '')
                    if action not in button_data:
                        button_data[action] = []
                    if value not in button_data[action]:
                        button_data[action].append(value)
        
        return button_data
    
    def _preserve_vote_buttons(self, soup: BeautifulSoup) -> None:
        """
        Preserve vote/action button content BEFORE buttons are stripped.
        
        Replaces vote buttons with text spans so the vote count appears in markdown.
        This maintains context association between products and their vote counts.
        
        Example transformations:
        - <button class="upvote">Upvote(290)</button> → <span>[UPVOTES: 290]</span>
        - <button>33</button> (in vote context) → <span>[VOTES: 33]</span>
        """
        # 1. Handle buttons with vote/upvote in class (e.g., Product Hunt)
        vote_buttons = soup.find_all('button', class_=re.compile(r'vote|upvote|like', re.IGNORECASE))
        for btn in vote_buttons:
            text = btn.get_text(strip=True)
            # Pattern: "Upvote(290)" or "Upvote (290)" or just a number
            number_match = re.search(r'(\d+(?:[\.,]\d+)?[kKmM]?)', text)
            if number_match:
                number = number_match.group(1)
                # Replace button with a visible span
                span = soup.new_tag('span')
                span.string = f' [UPVOTES: {number}] '
                btn.replace_with(span)
        
        # 2. Handle standalone number buttons (common in Product Hunt, Hacker News)
        # These are buttons containing just a number, usually the vote count
        buttons_to_replace = []
        for btn in soup.find_all('button'):
            text = btn.get_text(strip=True)
            # Button with just a number (1-4 digits, possibly with k/M suffix)
            if re.match(r'^\d{1,4}[kKmM]?$', text):
                # Check if button has an SVG sibling (vote icon)
                has_svg = btn.find('svg') is not None
                # Check if parent/ancestor has vote-related class
                parent = btn.parent
                ancestor_has_vote = False
                for _ in range(3):  # Check up to 3 levels up
                    if parent:
                        parent_classes = ' '.join(parent.get('class', [])).lower()
                        if any(word in parent_classes for word in ['vote', 'upvote', 'like', 'reaction', 'count', 'score']):
                            ancestor_has_vote = True
                            break
                        parent = parent.parent
                
                # If it looks like a vote button, mark for replacement
                if has_svg or ancestor_has_vote or int(re.search(r'\d+', text).group()) < 10000:
                    # Assume numeric buttons are vote counts on interactive sites
                    buttons_to_replace.append((btn, text))
        
        # Replace numeric buttons with vote labels
        for btn, text in buttons_to_replace:
            span = soup.new_tag('span')
            span.string = f' [VOTES: {text}] '
            try:
                btn.replace_with(span)
            except ValueError:
                pass  # Already replaced
        
        # 3. Handle buttons with "Upvote(N)" pattern (no space)
        for btn in soup.find_all('button'):
            text = btn.get_text(strip=True)
            if 'upvote' in text.lower():
                # Pattern: Upvote(290) or Upvote (290)
                number_match = re.search(r'\(?(\d+)\)?', text)
                if number_match:
                    number = number_match.group(1)
                    span = soup.new_tag('span')
                    span.string = f' [UPVOTES: {number}] '
                    try:
                        btn.replace_with(span)
                    except ValueError:
                        pass  # Already replaced
    
    def _find_label_for_input(self, soup: BeautifulSoup, input_elem) -> str:
        """Find the label text for an input element"""
        input_id = input_elem.get('id')
        
        if input_id:
            label = soup.find('label', {'for': input_id})
            if label:
                return label.get_text(strip=True)
        
        # Check if input is wrapped in label
        parent = input_elem.parent
        if parent and parent.name == 'label':
            # Get text that's not from the input itself
            text = parent.get_text(strip=True)
            return text
        
        return ""
    
    def _remove_noise_tags(self, soup: BeautifulSoup) -> None:
        """Remove noise tags after data extraction"""
        for tag_name in self.NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Now safe to remove forms (already extracted)
        for tag in soup.find_all(['form', 'button', 'select', 'input', 'textarea', 'label']):
            tag.decompose()
    
    def _remove_noise_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements with noise class/ID patterns"""
        for tag in soup.find_all(True):
            try:
                classes = tag.get('class', [])
                id_attr = tag.get('id', '')
                
                class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
                combined = (class_str + ' ' + id_attr).lower()
                
                if any(pattern in combined for pattern in self.NOISE_PATTERNS):
                    tag.decompose()
            except (AttributeError, TypeError):
                continue
    
    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """Remove HTML comments"""
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up markdown output"""
        # Remove excessive blank lines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Remove trailing whitespace
        markdown = re.sub(r'[ \t]+$', '', markdown, flags=re.MULTILINE)
        
        # Remove lines that are just whitespace
        markdown = re.sub(r'^\s+$', '', markdown, flags=re.MULTILINE)
        
        return markdown.strip()


def create_llm_context(extracted: ExtractedContent, fields: List[str]) -> str:
    """
    Create an LLM-friendly context combining markdown and structured data.
    
    Args:
        extracted: The ExtractedContent from hybrid extraction
        fields: The fields we're trying to extract
        
    Returns:
        Combined context string for LLM
    """
    parts = []
    
    # Add markdown content
    parts.append("# PAGE CONTENT\n")
    parts.append(extracted.markdown)
    
    # Add structured data hints
    metadata = extracted.get_metadata_summary()
    if metadata:
        parts.append("\n\n# ADDITIONAL DATA (check these for field values)\n")
        parts.append(metadata)
    
    # Add JSON-LD summary if relevant
    if extracted.json_ld:
        parts.append("\n\n# STRUCTURED DATA (JSON-LD)\n")
        for item in extracted.json_ld[:3]:  # Limit to 3 objects
            # Summarize the JSON-LD object
            summary = json.dumps(item, indent=2, default=str)[:2000]
            parts.append(summary)
    
    return "\n".join(parts)

