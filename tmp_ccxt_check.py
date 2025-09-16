import os, ccxt
proxy = os.environ.get(''HTTPS_PROXY'') or ''http://127.0.0.1:7890''
ex = ccxt.binanceusdm({
    ''enableRateLimit'': True,
    ''timeout'': 20000,
    ''proxies'': {''http'': proxy, ''https'': proxy},
})
try:
    mkts = ex.load_markets()
    print(''load_markets OK'', len(mkts))
except Exception as e:
    print(''load_markets ERR'', type(e).__name__, e)