import asyncio
from binance import AsyncClient
from dotenv import load_dotenv
import os

load_dotenv()

async def test():
    print("🔌 Connecting to Binance Testnet...")
    
    client = await AsyncClient.create(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_SECRET_KEY"),
        testnet=True
    )

    # Test 1: Server time
    server_time = await client.get_server_time()
    print(f"✅ Connected! Server time: {server_time}")

    # Test 2: Account balance
    account = await client.get_account()
    balances = [
        b for b in account['balances']
        if float(b['free']) > 0 or float(b['locked']) > 0
    ]
    print(f"✅ Account balances: {balances[:5]}")  # Show first 5 balances
    
    # Test 3: Fetch BTC price
    ticker = await client.get_symbol_ticker(symbol="BTCUSDT")
    print(f"✅ BTC Price: ${ticker['price']}")

    await client.close_connection()
    print("\n🎉 ALL TESTS PASSED! Connection successful!")

asyncio.run(test())
