import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_res_static():
    # Residential Proxy
    proxy_url = "http://brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2:REDACTED_PROXY_PASS@brd.superproxy.io:22225"
    proxies = {'http': proxy_url, 'https': proxy_url}
    
    url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-Steel-GNE27JYMFS/320244018"
    
    # Common headers for Home Depot
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    
    logger.info("--- Testing Residential Proxy in Static Mode ---")
    try:
        response = requests.get(url, proxies=proxies, headers=headers, verify=False, timeout=60)
        logger.info(f"Result: status={response.status_code}, length={len(response.text)}")
        
        if response.status_code == 200:
            logger.info("✅ SUCCESS: Got a 200 status with residential proxy!")
        else:
            logger.warning(f"⚠️ Still got status {response.status_code}")
            if len(response.text) > 5000:
                logger.info("But it still got substantial content.")
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_res_static()
