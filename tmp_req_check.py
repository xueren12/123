import os, requests
proxy = os.environ.get(''HTTPS_PROXY'') or ''http://127.0.0.1:7890''
proxies = {''http'': proxy, ''https'': proxy}
print(''PROXY='', proxy)
for u in [''https://fapi.binance.com/fapi/v1/ping'', ''https://www.okx.com/api/v5/public/time'']:
    try:
        r = requests.get(u, timeout=10, proxies=proxies)
        print(''OK'', u, r.status_code)
    except Exception as e:
        print(''ERR'', u, type(e).__name__, e)