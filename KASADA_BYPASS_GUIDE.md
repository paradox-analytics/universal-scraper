# Kasada Bypass Strategy with Bright Data

## The Challenge: Why Residential Proxies Aren't Enough
Kasada is a "zero-trust" anti-bot solution that goes beyond simple IP blocking. Even with high-quality residential proxies (like Bright Data), you can still be blocked because Kasada analyzes:

1.  **TLS Fingerprinting**: The cryptographic handshake your browser makes. Standard Python/Node HTTP clients and even some headless browsers have distinct "bot" fingerprints.
2.  **JavaScript Execution**: Kasada injects complex JS challenges (like `ips.js`) that check for automation flags (`navigator.webdriver`, `headless` properties).
3.  **Behavioral Biometrics**: Mouse movements, timing, and interaction patterns.
4.  **TCP/IP Fingerprinting**: Mismatches between your proxy's location and your browser's timezone/language settings.

## The Solution: A Layered Approach

### 1. The "Silver Bullet": Bright Data Web Unblocker
The most reliable "universal" solution is to offload the fingerprinting battle to Bright Data's **Web Unblocker** or **Scraping Browser**.

*   **How it works**: Instead of just routing traffic, their servers manage the browser fingerprint, solve the JS challenges (CAPTCHAs, Kasada proof-of-work), and rotate headers automatically.
*   **Configuration**:
    *   Change your proxy host from `brd.superproxy.io` to the Web Unblocker endpoint (usually specific to your zone).
    *   This effectively makes the "Fetcher" layer invisible to Kasada.

### 2. The "Hard Way": Advanced Camoufox Configuration
If you must use standard residential proxies with your own browser (Camoufox), you need to tune it perfectly:

#### A. Fingerprint Consistency
Ensure your browser fingerprint matches your proxy IP:
```python
# In CamoufoxFetcher
camoufox_config = {
    "geoip": True,  # Automatically adjust timezone/locale to match Proxy IP
    "os": "windows", # Force a common OS (Kasada suspects Linux/Headless)
    "screen": {"width": 1920, "height": 1080}, # Standard resolution
}
```

#### B. Header Rotation
Kasada flags static headers. You must rotate:
*   `User-Agent` (must match the browser version exactly)
*   `Accept-Language`
*   `Sec-Ch-Ua` (Client Hints)

#### C. Humanization
Enable maximum humanization in Camoufox:
```python
# In UniversalScraper
scraper = UniversalScraper(
    use_camoufox=True,
    fetch_mode='browser',
    # Camoufox specific internal flags
    humanize=True,
    stealth=True
)
```

### 3. Recommended Architecture for Chewy.com

1.  **Primary**: **Bright Data Web Unblocker** (Zone: `unblocker`)
    *   Success Rate: High (>95%)
    *   Cost: Higher per GB
    *   Complexity: Low (just a proxy URL change)

2.  **Secondary**: **Camoufox + Residential Proxies** (Zone: `residential`)
    *   Success Rate: Moderate (requires constant tuning)
    *   Cost: Lower
    *   Complexity: High (fingerprint management)

## Implementation in Universal Scraper

To implement the **Web Unblocker** approach, you simply need to update the `proxy_config` in your script:

```python
proxy_config = {
    'server': 'http://brd.superproxy.io:22225', # Web Unblocker port
    'username': 'brd-customer-hl_xxxx-zone-unblocker', # Unblocker zone
    'password': 'your_password'
}
```

No code changes are needed in `UniversalScraper` itself, as it passes this config through.
