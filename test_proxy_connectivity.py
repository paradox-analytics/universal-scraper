"""
Test Proxy and Web Unblocker Connectivity
Verifies that both residential proxy and Web Unblocker can connect successfully
"""
import asyncio
import logging
import requests
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_residential_proxy():
    """Test residential proxy connection"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Residential Proxy Connection")
    logger.info("="*60)
    
    # Credentials
    res_host = "brd.superproxy.io"
    res_port = "33335"
    res_user = "brd-customer-hl_803e8195-zone-residential_proxy2"
    res_pass = "rs2mvj79xi2t"
    
    # Build proxy URL
    safe_user = quote(res_user)
    safe_pass = quote(res_pass)
    proxy_url = f"http://{safe_user}:{safe_pass}@{res_host}:{res_port}"
    
    proxies = {
        "http": proxy_url,
        "https": proxy_url
    }
    
    logger.info(f"Proxy: {res_host}:{res_port}")
    logger.info(f"Zone: residential_proxy2")
    
    try:
        # Test with IP check
        logger.info("\n📡 Testing connection to https://api.ipify.org...")
        response = requests.get(
            "https://api.ipify.org?format=json",
            proxies=proxies,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            ip_data = response.json()
            logger.info(f"✅ Residential Proxy Connected Successfully!")
            logger.info(f"   Proxy IP: {ip_data.get('ip')}")
            logger.info(f"   Status Code: {response.status_code}")
            return True
        else:
            logger.error(f"❌ Residential Proxy Failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Residential Proxy Connection Failed: {e}")
        return False

def test_web_unblocker():
    """Test Web Unblocker connection"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Testing Web Unblocker Connection")
    logger.info("="*60)
    
    # Credentials
    unblocker_user = "brd-customer-hl_803e8195-zone-web_unlocker1"
    unblocker_pass = "t8mhp1qev1i1"
    unblocker_host = "brd.superproxy.io"
    unblocker_port = "33335"
    
    # Build proxy URL
    safe_user = quote(unblocker_user)
    safe_pass = quote(unblocker_pass)
    proxy_url = f"http://{safe_user}:{safe_pass}@{unblocker_host}:{unblocker_port}"
    
    proxies = {
        "http": proxy_url,
        "https": proxy_url
    }
    
    logger.info(f"Proxy: {unblocker_host}:{unblocker_port}")
    logger.info(f"Zone: web_unlocker1")
    
    try:
        # Test with IP check
        logger.info("\n📡 Testing connection to https://api.ipify.org...")
        response = requests.get(
            "https://api.ipify.org?format=json",
            proxies=proxies,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            ip_data = response.json()
            logger.info(f"✅ Web Unblocker Connected Successfully!")
            logger.info(f"   Proxy IP: {ip_data.get('ip')}")
            logger.info(f"   Status Code: {response.status_code}")
            
            # Test with a real website
            logger.info("\n📡 Testing with https://www.homedepot.com...")
            response2 = requests.get(
                "https://www.homedepot.com",
                proxies=proxies,
                verify=False,
                timeout=60
            )
            
            if response2.status_code == 200:
                logger.info(f"✅ Web Unblocker Successfully Fetched Home Depot!")
                logger.info(f"   Status Code: {response2.status_code}")
                logger.info(f"   Content Length: {len(response2.text)} bytes")
                return True
            else:
                logger.warning(f"⚠️ Home Depot returned status {response2.status_code}")
                return False
        else:
            logger.error(f"❌ Web Unblocker Failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Web Unblocker Connection Failed: {e}")
        return False

def main():
    """Run all connectivity tests"""
    logger.info("\n🚀 Starting Proxy Connectivity Tests\n")
    
    results = {
        "residential_proxy": test_residential_proxy(),
        "web_unblocker": test_web_unblocker()
    }
    
    logger.info("\n" + "="*60)
    logger.info("📊 Test Summary")
    logger.info("="*60)
    
    for name, success in results.items():
        icon = "✅" if success else "❌"
        logger.info(f"{icon} {name.replace('_', ' ').title()}: {'PASSED' if success else 'FAILED'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n✅ All connectivity tests passed! Ready for adaptive testing.")
    else:
        logger.error("\n❌ Some connectivity tests failed. Fix these before running adaptive tests.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
