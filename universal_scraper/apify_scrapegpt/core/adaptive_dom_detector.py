"""
Adaptive DOM Pattern Detector with Reinforcement Learning-Style Iteration

Multi-pass approach:
1. Fast content-based detection (no LLM)
2. LLM-guided nested structure analysis (if quality < threshold)
3. Deep context analysis with error feedback (if still failing)

This ensures we find the correct selectors even on challenging websites.
"""

import logging
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup, Tag
import litellm

logger = logging.getLogger(__name__)


class AdaptiveDOMDetector:
    """
    Adaptive DOM detector with reinforcement-style iteration.
    
    Automatically retries with deeper analysis if initial detection fails.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gpt-4o-mini',
        max_passes: int = 3
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_passes = max_passes
        
    def detect_with_reinforcement(
        self,
        html: str,
        fields: List[str],
        initial_pattern: Optional[Dict[str, Any]],
        extraction_result: Optional[Dict[str, Any]] = None,
        pass_number: int = 1
    ) -> Dict[str, Any]:
        """
        Detect DOM pattern with reinforcement iteration.
        
        Args:
            html: Page HTML
            fields: Fields to extract
            initial_pattern: Pattern from fast content-based detection
            extraction_result: Result from extraction attempt (items, quality)
            pass_number: Current iteration (1-3)
            
        Returns:
            Improved pattern with higher confidence selectors
        """
        
        # Pass 1: Use initial content-based detection
        if pass_number == 1:
            logger.info(" Pass 1: Using content-based detection")
            return initial_pattern or {}
        
        # Pass 2: LLM-guided nested structure analysis
        if pass_number == 2:
            logger.info(" Pass 2: LLM-guided nested structure analysis")
            return self._llm_analyze_nested_structures(
                html=html,
                fields=fields,
                failed_pattern=initial_pattern,
                extraction_result=extraction_result
            )
        
        # Pass 3: Deep context analysis with error feedback
        if pass_number == 3:
            logger.info(" Pass 3: Deep context analysis with error feedback")
            return self._llm_deep_context_analysis(
                html=html,
                fields=fields,
                failed_patterns=[initial_pattern],
                extraction_history=extraction_result
            )
        
        return initial_pattern or {}
    
    def _llm_analyze_nested_structures(
        self,
        html: str,
        fields: List[str],
        failed_pattern: Optional[Dict[str, Any]],
        extraction_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ask LLM to analyze why selectors failed and suggest better patterns.
        
        This is the REINFORCEMENT step - learn from failure.
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Build context about what failed
        failure_context = self._build_failure_context(failed_pattern, extraction_result)
        
        # Extract relevant HTML sample (around failed selector if available)
        html_sample = self._extract_relevant_html_sample(soup, failed_pattern)
        
        # Prompt LLM to analyze nested structures
        prompt = f"""You are analyzing a website's HTML to find the correct data container pattern.

**CONTEXT:**
{failure_context}

**TASK:**
Analyze the HTML below and identify the CORRECT repeating element that contains the data.
**CRITICAL**: Focus on fixing the fields that are currently NULL!

**FIELDS TO EXTRACT:**
{', '.join(fields)}

**WHAT TO LOOK FOR:**
1. Find elements that repeat 10-50 times (data containers, not UI)
2. Each element should contain ALL or MOST of the requested fields
3. Look for nested structures (data might be 2-3 levels deep)
4. Check both CSS classes AND tag hierarchy (article > div > h3)
5. Ignore navigation, filters, and UI elements

**CRITICAL: DATA MAY BE IN SIBLING ELEMENTS, NOT JUST CHILDREN!**

Many websites (Stack Overflow, GitHub, Indeed) use sibling-based layouts where related
data is in ADJACENT elements, not nested inside the container.

Example (Stack Overflow):
```html
<div class="s-post-summary">  ← Main container
    <h3>Question Title</h3>    Inside container
</div>
<div class="s-post-summary--stats">  ← SIBLING (not child!)
    <span class="vote-count">42</span>  ← votes field is HERE
</div>
```

**WHERE TO LOOK FOR NULL FIELDS:**
1.  Inside the container (children, grandchildren)
2.  **SIBLING elements** (next/previous siblings of the container)
3.  Parent element (shared across all items)
4. Check attributes (data-*, aria-*, title, datetime) not just text content
5. Numbers might be in <span>, <a>, or <button> tags
6. Dates might be in <time> tags with datetime attribute

**HTML SAMPLE:**
```html
{html_sample[:15000]}
```

**RESPOND WITH JSON:**
{{
    "selector": "CSS selector for repeating container (e.g., 'article.item', 'div.s-post-summary')",
    "count_estimate": "estimated number of elements",
    "confidence": "0.0-1.0",
    "reasoning": "why this selector is correct",
    "nested_hints": {{
        "field_name": "EXACT CSS selector from container (e.g., 'span.vote-count-post')"
    }}
}}

**CRITICAL**: Provide nested_hints for EVERY field, especially the NULL ones!
"""
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            import json
            analysis = json.loads(result)
            
            logger.info(f"    LLM suggested: {analysis.get('selector')} (confidence: {analysis.get('confidence')})")
            logger.info(f"    Reasoning: {analysis.get('reasoning', 'N/A')[:100]}")
            
            return {
                'type': 'llm_guided',
                'selector': analysis.get('selector'),
                'confidence': float(analysis.get('confidence', 0.7)),
                'count': int(analysis.get('count_estimate', 10)),
                'nested_hints': analysis.get('nested_hints', {}),
                'reasoning': analysis.get('reasoning'),
                'pass': 2
            }
            
        except Exception as e:
            logger.error(f"    LLM analysis failed: {e}")
            return failed_pattern or {}
    
    def _llm_deep_context_analysis(
        self,
        html: str,
        fields: List[str],
        failed_patterns: List[Dict],
        extraction_history: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Deep LLM analysis with full error feedback.
        
        This is the FINAL ATTEMPT - use maximum context.
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Build comprehensive failure report
        failure_report = self._build_comprehensive_failure_report(
            failed_patterns, 
            extraction_history
        )
        
        # Extract larger HTML sample (more context)
        html_sample = html[:30000]  # 30KB sample
        
        prompt = f"""You are a web scraping expert analyzing a challenging website.

**SITUATION:**
Multiple attempts to find the correct data container have failed.

**FAILURE HISTORY:**
{failure_report}

**FIELDS TO EXTRACT:**
{', '.join(fields)}

**YOUR TASK:**
Perform DEEP analysis to find the correct pattern. Consider:

1. **Shadow DOM / Web Components**: Look for custom elements (e.g., <my-card>)
2. **Deeply Nested Structures**: Data might be 3-5 levels deep
3. **Dynamic Content**: Data loaded via JS (look for data-* attributes)
4. **Unconventional Patterns**: Grid layouts, flex containers, table structures
5. **Multiple Container Types**: Maybe data is split across different elements

**HTML SAMPLE (First 30KB):**
```html
{html_sample}
```

**RESPOND WITH JSON:**
{{
    "selector": "EXACT CSS selector",
    "alternative_selectors": ["fallback selector 1", "fallback selector 2"],
    "confidence": "0.0-1.0",
    "reasoning": "detailed explanation of why this is correct",
    "extraction_strategy": "nested_elements | attributes | mixed",
    "field_hints": {{
        "field_name": {{
            "selector": "relative CSS path",
            "attribute": "optional - if data is in attribute",
            "fallback": "alternative way to get this field"
        }}
    }}
}}
"""
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            import json
            analysis = json.loads(result)
            
            logger.info(f"    LLM DEEP analysis: {analysis.get('selector')} (confidence: {analysis.get('confidence')})")
            logger.info(f"    Strategy: {analysis.get('extraction_strategy')}")
            logger.info(f"    Reasoning: {analysis.get('reasoning', 'N/A')[:150]}")
            
            return {
                'type': 'llm_deep_analysis',
                'selector': analysis.get('selector'),
                'alternative_selectors': analysis.get('alternative_selectors', []),
                'confidence': float(analysis.get('confidence', 0.8)),
                'extraction_strategy': analysis.get('extraction_strategy'),
                'field_hints': analysis.get('field_hints', {}),
                'reasoning': analysis.get('reasoning'),
                'pass': 3
            }
            
        except Exception as e:
            logger.error(f"    Deep LLM analysis failed: {e}")
            return failed_patterns[0] if failed_patterns else {}
    
    def _build_failure_context(
        self, 
        failed_pattern: Optional[Dict[str, Any]],
        extraction_result: Optional[Dict[str, Any]]
    ) -> str:
        """Build context string explaining what failed"""
        
        if not extraction_result:
            return "Initial detection failed - no pattern found"
        
        items = extraction_result.get('items', [])
        quality = extraction_result.get('quality', 0)
        
        context = []
        
        if failed_pattern:
            context.append(f"Tried selector: {failed_pattern.get('selector')}")
            context.append(f"Pattern type: {failed_pattern.get('type')}")
        
        context.append(f"Items extracted: {len(items)}")
        context.append(f"Quality: {quality:.0f}%")
        
        if len(items) == 0:
            context.append(" PROBLEM: 0 items extracted - selector is wrong")
        elif quality < 70:
            # Analyze per-field quality across ALL items
            if items:
                field_quality = {}
                for field in items[0].keys():
                    filled = sum(1 for item in items if item.get(field) not in (None, '', []))
                    field_quality[field] = (filled / len(items)) * 100
                
                # Sort by quality (worst first)
                worst_fields = sorted(field_quality.items(), key=lambda x: x[1])
                
                context.append(f" PROBLEM: {quality:.0f}% quality - Per-field analysis:")
                for field, field_qual in worst_fields[:5]:  # Show worst 5 fields
                    status = "" if field_qual >= 80 else "" if field_qual >= 50 else ""
                    context.append(f"   {status} {field}: {field_qual:.0f}% filled")
                
                # Highlight critical issues
                null_fields = [f for f, q in worst_fields if q == 0]
                if null_fields:
                    context.append(f" CRITICAL: These fields are ALWAYS null: {', '.join(null_fields)}")
                    context.append("   → The CSS selectors for these fields are likely incorrect!")
        
        return '\n'.join(context)
    
    def _build_comprehensive_failure_report(
        self,
        failed_patterns: List[Dict],
        extraction_history: Optional[Dict[str, Any]]
    ) -> str:
        """Build detailed report of all failed attempts"""
        
        report = []
        
        for i, pattern in enumerate(failed_patterns, 1):
            report.append(f"\n**Attempt {i}:**")
            report.append(f"  Selector: {pattern.get('selector', 'N/A')}")
            report.append(f"  Type: {pattern.get('type', 'N/A')}")
            report.append(f"  Confidence: {pattern.get('confidence', 0):.2f}")
            
            if extraction_history:
                items = extraction_history.get('items', [])
                quality = extraction_history.get('quality', 0)
                report.append(f"  Result: {len(items)} items, {quality:.0f}% quality")
        
        if not failed_patterns:
            report.append("No patterns were found in previous attempts")
        
        return '\n'.join(report)
    
    def _extract_relevant_html_sample(
        self,
        soup: BeautifulSoup,
        failed_pattern: Optional[Dict[str, Any]],
        sample_size: int = 15000
    ) -> str:
        """
        Extract relevant HTML section (around failed selector if available)
        """
        
        if not failed_pattern or not failed_pattern.get('selector'):
            # No context, return body content
            body = soup.find('body')
            if body:
                return str(body)[:sample_size]
            return str(soup)[:sample_size]
        
        try:
            # Try to find parent container of failed selector
            selector = failed_pattern['selector']
            elements = soup.select(selector, limit=3)
            
            if elements:
                # Get parent container
                parent = elements[0].parent
                if parent:
                    return str(parent)[:sample_size]
            
            # Fallback: return body
            body = soup.find('body')
            if body:
                return str(body)[:sample_size]
            
        except Exception as e:
            logger.debug(f"Error extracting relevant HTML: {e}")
        
        return str(soup)[:sample_size]

