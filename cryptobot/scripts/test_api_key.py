import ccxt.async_support as ccxt_async
import asyncio

async def test_mexc_api():
    api_key = 'mx0vglqB4WmkK4Qv0G'
    api_secret = '32686a1469404ecc8b128d12dba6433b'
    
    exchange = ccxt_async.mexc({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'timeout': 30000,
    })
    
    try:
        # Test fetching markets or a simple authenticated endpoint
        markets = await exchange.load_markets()
        print("Markets loaded successfully:", markets.keys())
        
        # Test fetching OHLCV data
        ohlcv = await exchange.fetch_ohlcv('PI/USDT', '5m', limit=10)
        print("OHLCV data:", ohlcv)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(test_mexc_api())