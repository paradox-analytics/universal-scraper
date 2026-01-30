import requests
import logging
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

    # Load from .env
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    proxy_url_env = os.getenv("BRIGHT_DATA_PROXY_URL")
    if not proxy_url_env:
        logger.error("❌ BRIGHT_DATA_PROXY_URL not found in .env")
        return

    logger.info(f"Testing proxy URL from .env: {proxy_url_env.split('@')[1] if '@' in proxy_url_env else '***'}")
    
    proxies = {
        "http": proxy_url_env,
        "https": proxy_url_env
    }
    
    try:
        logger.info("Sending request to https://api.ipify.org?format=json")
        response = requests.get(
            "https://api.ipify.org?format=json",
            proxies=proxies,
            verify=False, # Disable SSL verification as HTMLFetcher does
            timeout=30
        )
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Proxy connection successful!")
        else:
            logger.error("❌ Proxy connection failed with status code.")
            
    except Exception as e:
        logger.error(f"❌ Proxy connection failed with exception: {e}")

if __name__ == "__main__":
    test_proxy_connection()
