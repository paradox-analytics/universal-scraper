"""
Structural Embedding Generator - Creates embeddings from HTML structure

This module generates vector representations of HTML structure (not content)
to enable similarity matching between websites. Similar websites (e.g., e-commerce sites)
will have similar embeddings, allowing pattern reuse.

Key Insight: Most websites fall into ~100 structural patterns (e-commerce, news, forums, etc.)
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class StructuralEmbedding:
    """
    Generate structural embeddings from HTML.
    
    Unlike content embeddings (which encode meaning), structural embeddings
    encode the HTML structure, layout patterns, and semantic markup.
    
    This enables finding websites with similar structure, which often
    have similar data extraction patterns.
    """
    
    # Common HTML tags to analyze (prioritized by importance)
    SEMANTIC_TAGS = [
        'article', 'section', 'nav', 'aside', 'header', 'footer', 'main',
        'figure', 'figcaption', 'time', 'mark', 'details', 'summary'
    ]
    
    CONTENT_TAGS = [
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'div', 'a',
        'strong', 'em', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th',
        'form', 'input', 'button', 'label', 'select', 'textarea'
    ]
    
    MEDIA_TAGS = ['img', 'video', 'audio', 'picture', 'source', 'svg', 'canvas']
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize structural embedding generator.
        
        Args:
            embedding_dim: Dimension of output embedding (default: 512)
        """
        self.embedding_dim = embedding_dim
        logger.info(f" Structural Embedding initialized (dim={embedding_dim})")
    
    def generate(self, html: str) -> np.ndarray:
        """
        Generate structural embedding from HTML.
        
        Args:
            html: Raw HTML string
            
        Returns:
            NumPy array of shape (embedding_dim,) representing HTML structure
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structural features
        features = []
        
        # 1. Tag Frequency Features (100 dims)
        tag_freq = self._extract_tag_frequencies(soup)
        features.extend(tag_freq)
        
        # 2. Depth & Nesting Features (50 dims)
        depth_features = self._extract_depth_features(soup)
        features.extend(depth_features)
        
        # 3. Attribute Patterns (100 dims)
        attr_features = self._extract_attribute_features(soup)
        features.extend(attr_features)
        
        # 4. Structural Patterns (100 dims)
        struct_features = self._extract_structural_patterns(soup)
        features.extend(struct_features)
        
        # 5. Content Density Features (50 dims)
        density_features = self._extract_density_features(soup)
        features.extend(density_features)
        
        # 6. Semantic Markup Features (50 dims)
        semantic_features = self._extract_semantic_features(soup)
        features.extend(semantic_features)
        
        # 7. Layout Indicators (62 dims)
        layout_features = self._extract_layout_features(soup)
        features.extend(layout_features)
        
        # Normalize to embedding_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.embedding_dim:
            # Pad with zeros
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        elif len(features) > self.embedding_dim:
            # Truncate
            features = features[:self.embedding_dim]
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        logger.debug(f"Generated embedding: dim={len(features)}, norm={np.linalg.norm(features):.4f}")
        return features
    
    def _extract_tag_frequencies(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract normalized tag frequencies.
        
        Returns 100-dim feature vector of tag frequencies.
        """
        all_tags = [tag.name for tag in soup.find_all()]
        tag_counts = Counter(all_tags)
        total_tags = len(all_tags)
        
        if total_tags == 0:
            return [0.0] * 100
        
        features = []
        
        # Semantic tags (20 features)
        for tag in self.SEMANTIC_TAGS:
            freq = tag_counts.get(tag, 0) / total_tags
            features.append(freq)
        # Pad to 20
        features.extend([0.0] * (20 - len(features)))
        
        # Content tags (40 features)
        content_features = []
        for tag in self.CONTENT_TAGS:
            freq = tag_counts.get(tag, 0) / total_tags
            content_features.append(freq)
        # Pad to 40
        content_features.extend([0.0] * (40 - len(content_features)))
        features.extend(content_features[:40])
        
        # Media tags (10 features)
        media_features = []
        for tag in self.MEDIA_TAGS:
            freq = tag_counts.get(tag, 0) / total_tags
            media_features.append(freq)
        # Pad to 10
        media_features.extend([0.0] * (10 - len(media_features)))
        features.extend(media_features[:10])
        
        # Top 30 most common tags (general)
        most_common = tag_counts.most_common(30)
        for _, count in most_common:
            features.append(count / total_tags)
        # Pad to 30
        features.extend([0.0] * (30 - len(most_common)))
        
        return features[:100]
    
    def _extract_depth_features(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract DOM tree depth and nesting features.
        
        Returns 50-dim feature vector.
        """
        depths = []
        for tag in soup.find_all():
            depth = len(list(tag.parents))
            depths.append(depth)
        
        if not depths:
            return [0.0] * 50
        
        features = []
        
        # Basic statistics (5 features)
        features.append(np.mean(depths))
        features.append(np.std(depths))
        features.append(np.min(depths))
        features.append(np.max(depths))
        features.append(np.median(depths))
        
        # Depth distribution (20 bins)
        max_depth = max(depths)
        if max_depth > 0:
            hist, _ = np.histogram(depths, bins=20, range=(0, max_depth))
            hist = hist / len(depths)
            features.extend(hist.tolist())
        else:
            features.extend([0.0] * 20)
        
        # Repeating depth patterns (15 features)
        depth_counter = Counter(depths)
        most_common_depths = depth_counter.most_common(15)
        for depth, count in most_common_depths:
            features.append(count / len(depths))
        # Pad to 15
        features.extend([0.0] * (15 - len(most_common_depths)))
        
        # Container nesting (10 features)
        div_depths = [len(list(tag.parents)) for tag in soup.find_all('div')]
        if div_depths:
            features.extend([
                np.mean(div_depths),
                np.std(div_depths),
                np.max(div_depths),
                len(div_depths) / len(depths),  # div ratio
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
            ])
        else:
            features.extend([0.0] * 10)
        
        return features[:50]
    
    def _extract_attribute_features(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract attribute pattern features (classes, ids, data-*, aria-*, etc).
        
        Returns 100-dim feature vector.
        """
        features = []
        
        # Count elements with specific attribute patterns
        all_elements = soup.find_all()
        total_elements = len(all_elements)
        
        if total_elements == 0:
            return [0.0] * 100
        
        # Data attributes (20 features)
        data_attr_patterns = [
            'data-id', 'data-name', 'data-value', 'data-price', 'data-title',
            'data-url', 'data-src', 'data-href', 'data-type', 'data-target',
            'data-product', 'data-item', 'data-testid', 'data-cy', 'data-qa',
            'data-component', 'data-index', 'data-key', 'data-label', 'data-role'
        ]
        for pattern in data_attr_patterns:
            count = len([e for e in all_elements if e.has_attr(pattern)])
            features.append(count / total_elements)
        
        # ARIA attributes (15 features)
        aria_patterns = [
            'aria-label', 'aria-labelledby', 'aria-describedby', 'aria-hidden',
            'aria-expanded', 'aria-selected', 'aria-checked', 'aria-disabled',
            'aria-required', 'aria-live', 'aria-atomic', 'aria-relevant',
            'aria-controls', 'aria-owns', 'aria-haspopup'
        ]
        for pattern in aria_patterns:
            count = len([e for e in all_elements if e.has_attr(pattern)])
            features.append(count / total_elements)
        
        # Class patterns (30 features)
        all_classes = []
        for elem in all_elements:
            if elem.has_attr('class'):
                all_classes.extend(elem['class'])
        
        class_counter = Counter(all_classes)
        # Check for common class naming patterns
        class_patterns = {
            'container': r'container|wrapper|wrap',
            'card': r'card|item|box',
            'grid': r'grid|row|col',
            'list': r'list|menu|nav',
            'button': r'btn|button',
            'header': r'header|head|title',
            'footer': r'footer|foot',
            'content': r'content|body|main',
            'sidebar': r'sidebar|aside',
            'form': r'form|input',
            'modal': r'modal|dialog|popup',
            'image': r'image|img|pic|photo',
            'text': r'text|label|caption',
            'link': r'link|anchor',
            'icon': r'icon|svg'
        }
        
        for pattern_name, pattern_regex in class_patterns.items():
            count = sum(1 for cls in all_classes if re.search(pattern_regex, cls.lower()))
            features.append(count / max(len(all_classes), 1))
        
        # Pad to 30
        features.extend([0.0] * (30 - len(class_patterns)))
        
        # ID patterns (10 features)
        ids = [elem.get('id', '') for elem in all_elements if elem.has_attr('id')]
        features.extend([
            len(ids) / total_elements,  # ID usage ratio
            len(set(ids)) / max(len(ids), 1),  # ID uniqueness
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Role attributes (10 features)
        role_values = ['navigation', 'main', 'complementary', 'banner', 'contentinfo',
                       'search', 'form', 'article', 'region', 'alert']
        for role in role_values:
            count = len(soup.find_all(attrs={'role': role}))
            features.append(count / total_elements)
        
        # Microdata (5 features)
        itemscope_count = len(soup.find_all(attrs={'itemscope': True}))
        itemprop_count = len(soup.find_all(attrs={'itemprop': True}))
        itemtype_count = len(soup.find_all(attrs={'itemtype': True}))
        features.extend([
            itemscope_count / total_elements,
            itemprop_count / total_elements,
            itemtype_count / total_elements,
            0.0, 0.0  # padding
        ])
        
        # JSON-LD indicators (10 features)
        json_ld_scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
        features.extend([
            len(json_ld_scripts) / max(total_elements, 1),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        return features[:100]
    
    def _extract_structural_patterns(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract repeating structural patterns (key for data extraction).
        
        Returns 100-dim feature vector.
        """
        features = []
        
        # Find repeating element signatures
        all_elements = soup.find_all()
        
        if not all_elements:
            return [0.0] * 100
        
        # Signature: tag + class names
        signatures = []
        for elem in all_elements:
            sig = elem.name
            if elem.has_attr('class'):
                sig += '.' + '.'.join(sorted(elem['class'][:3]))  # Top 3 classes
            signatures.append(sig)
        
        sig_counter = Counter(signatures)
        
        # Repeating patterns (30 features)
        most_repeated = sig_counter.most_common(30)
        for sig, count in most_repeated:
            if count > 1:  # Only repeating elements
                features.append(count / len(signatures))
        # Pad to 30
        features.extend([0.0] * (30 - len(most_repeated)))
        
        # Sibling patterns (20 features)
        sibling_patterns = Counter()
        for elem in all_elements:
            siblings = [s.name for s in elem.find_next_siblings(limit=5)]
            if siblings:
                pattern = ','.join(siblings)
                sibling_patterns[pattern] += 1
        
        top_sibling = sibling_patterns.most_common(20)
        for pattern, count in top_sibling:
            features.append(count / len(all_elements))
        # Pad to 20
        features.extend([0.0] * (20 - len(top_sibling)))
        
        # Parent-child patterns (20 features)
        parent_child = Counter()
        for elem in all_elements:
            if elem.parent:
                pattern = f"{elem.parent.name}>{elem.name}"
                parent_child[pattern] += 1
        
        top_parent_child = parent_child.most_common(20)
        for pattern, count in top_parent_child:
            features.append(count / len(all_elements))
        # Pad to 20
        features.extend([0.0] * (20 - len(top_parent_child)))
        
        # List patterns (15 features)
        ul_count = len(soup.find_all('ul'))
        ol_count = len(soup.find_all('ol'))
        li_count = len(soup.find_all('li'))
        features.extend([
            ul_count / len(all_elements),
            ol_count / len(all_elements),
            li_count / len(all_elements),
            li_count / max(ul_count + ol_count, 1),  # avg items per list
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Table patterns (15 features)
        table_count = len(soup.find_all('table'))
        tr_count = len(soup.find_all('tr'))
        td_count = len(soup.find_all('td'))
        features.extend([
            table_count / len(all_elements),
            tr_count / len(all_elements),
            td_count / len(all_elements),
            td_count / max(tr_count, 1),  # avg cells per row
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        return features[:100]
    
    def _extract_density_features(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract content density features (text vs markup ratio).
        
        Returns 50-dim feature vector.
        """
        features = []
        
        # Text content analysis
        all_text = soup.get_text()
        text_length = len(all_text)
        html_length = len(str(soup))
        
        # Basic density (10 features)
        features.extend([
            text_length / max(html_length, 1),  # text density
            html_length / 1000.0,  # normalized HTML size
            text_length / 1000.0,  # normalized text size
            len(soup.find_all()) / max(text_length, 1),  # elements per char
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Link density (10 features)
        links = soup.find_all('a')
        link_text_length = sum(len(link.get_text()) for link in links)
        features.extend([
            len(links) / max(len(soup.find_all()), 1),  # link ratio
            link_text_length / max(text_length, 1),  # link text ratio
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Image density (10 features)
        images = soup.find_all('img')
        features.extend([
            len(images) / max(len(soup.find_all()), 1),  # image ratio
            len(images) / max(text_length / 1000, 1),  # images per 1k chars
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Heading density (10 features)
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading_text = sum(len(h.get_text()) for h in headings)
        features.extend([
            len(headings) / max(len(soup.find_all()), 1),  # heading ratio
            heading_text / max(text_length, 1),  # heading text ratio
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Form density (10 features)
        forms = soup.find_all('form')
        inputs = soup.find_all(['input', 'textarea', 'select', 'button'])
        features.extend([
            len(forms) / max(len(soup.find_all()), 1),  # form ratio
            len(inputs) / max(len(soup.find_all()), 1),  # input ratio
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        return features[:50]
    
    def _extract_semantic_features(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract semantic HTML5 features.
        
        Returns 50-dim feature vector.
        """
        features = []
        all_elements = soup.find_all()
        total = len(all_elements)
        
        if total == 0:
            return [0.0] * 50
        
        # Semantic HTML5 tags (20 features)
        for tag in self.SEMANTIC_TAGS:
            count = len(soup.find_all(tag))
            features.append(count / total)
        # Pad to 20
        features.extend([0.0] * (20 - len(self.SEMANTIC_TAGS)))
        
        # Schema.org microdata (10 features)
        features.extend([
            len(soup.find_all(attrs={'itemscope': True})) / total,
            len(soup.find_all(attrs={'itemprop': True})) / total,
            len(soup.find_all(attrs={'itemtype': True})) / total,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # Open Graph / Meta tags (10 features)
        og_meta = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
        twitter_meta = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
        features.extend([
            len(og_meta) / max(total, 1),
            len(twitter_meta) / max(total, 1),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        # WAI-ARIA landmarks (10 features)
        landmarks = soup.find_all(attrs={'role': re.compile(r'(navigation|main|banner|contentinfo|search|form|complementary)')})
        features.extend([
            len(landmarks) / total,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
        ])
        
        return features[:50]
    
    def _extract_layout_features(self, soup: BeautifulSoup) -> List[float]:
        """
        Extract layout/design indicators with improved domain-specific patterns.
        
        Returns 62-dim feature vector.
        """
        features = []
        all_elements = soup.find_all()
        total = len(all_elements)
        
        if total == 0:
            return [0.0] * 62
        
        # Helper function for better pattern matching
        def count_pattern(elements, patterns, check_class=True, check_id=True, check_attrs=True):
            """Count elements matching patterns in class, id, or attributes."""
            count = 0
            for elem in elements:
                # Check classes
                if check_class and elem.has_attr('class'):
                    classes = ' '.join(elem.get('class', [])).lower()
                    if any(re.search(pattern, classes) for pattern in patterns):
                        count += 1
                        continue
                
                # Check IDs
                if check_id and elem.has_attr('id'):
                    elem_id = elem.get('id', '').lower()
                    if any(re.search(pattern, elem_id) for pattern in patterns):
                        count += 1
                        continue
                
                # Check data-* and other attributes
                if check_attrs:
                    for attr, value in elem.attrs.items():
                        if isinstance(value, str):
                            if any(re.search(pattern, value.lower()) for pattern in patterns):
                                count += 1
                                break
            return count
        
        # E-commerce indicators (20 features) - ENHANCED
        ecommerce_patterns = [
            r'\bcart\b', r'\bcheckout\b', r'\bproduct\b', r'\bprice\b', 
            r'\bbuy\b', r'\bshop\b', r'\bitem\b', r'\badd.?to.?cart\b',
            r'\bwishlist\b', r'\bfavorite\b', r'\bpayment\b', r'\bshipping\b',
            r'\border\b', r'\bsku\b', r'\binventory\b', r'\bdeal\b',
            r'\bsale\b', r'\bdiscount\b', r'\brating\b', r'\breview\b'
        ]
        
        ecommerce_count = count_pattern(all_elements, ecommerce_patterns)
        ecommerce_score = ecommerce_count / max(total, 1)
        
        # Detailed e-commerce sub-features
        cart_count = count_pattern(all_elements, [r'\bcart\b', r'\bbag\b', r'\bbasket\b'])
        price_count = count_pattern(all_elements, [r'\bprice\b', r'\bcost\b', r'\bamount\b'])
        product_count = count_pattern(all_elements, [r'\bproduct\b', r'\bitem\b'])
        rating_count = count_pattern(all_elements, [r'\brating\b', r'\bstar\b', r'\breview\b'])
        checkout_count = count_pattern(all_elements, [r'\bcheckout\b', r'\bpayment\b'])
        
        features.extend([
            ecommerce_score * 10,  # Amplify signal
            cart_count / max(total, 1),
            price_count / max(total, 1),
            product_count / max(total, 1),
            rating_count / max(total, 1),
            checkout_count / max(total, 1),
            len(soup.find_all(attrs={'itemprop': 'price'})) / max(total, 1),
            len(soup.find_all(attrs={'itemprop': 'product'})) / max(total, 1),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        # Forum/Community indicators (20 features) - ENHANCED
        forum_patterns = [
            r'\bpost\b', r'\bcomment\b', r'\breply\b', r'\bthread\b',
            r'\bvote\b', r'\bupvote\b', r'\bdownvote\b', r'\bscore\b',
            r'\bauthor\b', r'\buser\b', r'\bmember\b', r'\bdiscussion\b',
            r'\bforum\b', r'\bcommunity\b', r'\btopic\b', r'\bmessage\b',
            r'\bpoints\b', r'\bkarma\b', r'\breputation\b', r'\banswer\b'
        ]
        
        forum_count = count_pattern(all_elements, forum_patterns)
        forum_score = forum_count / max(total, 1)
        
        # Detailed forum sub-features
        post_count = count_pattern(all_elements, [r'\bpost\b', r'\bsubmission\b'])
        comment_count = count_pattern(all_elements, [r'\bcomment\b', r'\breply\b'])
        vote_count = count_pattern(all_elements, [r'\bvote\b', r'\bupvote\b', r'\bscore\b', r'\bpoints\b'])
        author_count = count_pattern(all_elements, [r'\bauthor\b', r'\buser\b', r'\bby\b'])
        thread_count = count_pattern(all_elements, [r'\bthread\b', r'\btopic\b', r'\bdiscussion\b'])
        
        features.extend([
            forum_score * 10,  # Amplify signal
            post_count / max(total, 1),
            comment_count / max(total, 1),
            vote_count / max(total, 1),
            author_count / max(total, 1),
            thread_count / max(total, 1),
            len(soup.find_all(attrs={'class': re.compile(r'comment|reply')})) / max(total, 1),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        # News/Blog indicators (22 features) - ENHANCED
        news_patterns = [
            r'\barticle\b', r'\bpost\b', r'\bblog\b', r'\bnews\b',
            r'\bstory\b', r'\bauthor\b', r'\bdate\b', r'\bpublish\b',
            r'\bcontent\b', r'\bheadline\b', r'\bbody\b', r'\bparagraph\b',
            r'\bcategory\b', r'\btag\b', r'\barchive\b', r'\bentry\b',
            r'\bjournalism\b', r'\breporter\b', r'\bbyline\b', r'\barticle\b'
        ]
        
        news_count = count_pattern(all_elements, news_patterns)
        news_score = news_count / max(total, 1)
        
        # Detailed news sub-features
        article_count = len(soup.find_all('article'))
        time_count = len(soup.find_all('time'))
        author_meta = len(soup.find_all('meta', attrs={'name': re.compile(r'author', re.I)}))
        publish_meta = len(soup.find_all('meta', attrs={'property': re.compile(r'published|article:published', re.I)}))
        headline_count = len(soup.find_all(['h1', 'h2'], attrs={'class': re.compile(r'headline|title', re.I)}))
        
        features.extend([
            news_score * 10,  # Amplify signal
            article_count / max(total, 1),
            time_count / max(total, 1),
            author_meta / max(total, 1),
            publish_meta / max(total, 1),
            headline_count / max(total, 1),
            len(soup.find_all(attrs={'itemprop': 'articleBody'})) / max(total, 1),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        return features[:62]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Clip to [0, 1] range
        return max(0.0, min(1.0, similarity))

