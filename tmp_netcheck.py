import requests, sys
urls = [''https://fapi.binance.com/fapi/v1/ping'', ''https://www.okx.com/api/v5/public/time'']
for u in urls:
    try:
        r = requests.get(u, timeout=5)
        print('OK', u, r.status_code)
    except Exception as e:
        print('ERR', u, type(e).__name__, e)