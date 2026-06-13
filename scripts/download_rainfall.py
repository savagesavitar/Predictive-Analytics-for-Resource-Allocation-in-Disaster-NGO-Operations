import requests, json

# Check India WRIS API for rainfall data resources
url = 'https://ckandev.indiadataportal.com/api/3/action/package_show?id=25664936-d131-4b07-845e-4bd2c0d794aa'
r = requests.get(url, timeout=30)
if r.status_code == 200:
    data = r.json()
    for resource in data['result']['resources']:
        name = resource.get('name', '')
        fmt = resource.get('format', '')
        rurl = resource.get('url', '')
        print(f'{name} [{fmt}]: {rurl[:100]}')
else:
    print(f'FAIL: {r.status_code}')

# Also try the direct resource download
res_url = 'https://ckandev.indiadataportal.com/dataset/25664936-d131-4b07-845e-4bd2c0d794aa/resource/d97707e6-9309-46a5-ac31-e8482ce8f9eb/download'
r2 = requests.get(res_url, timeout=120, headers={'User-Agent': 'Mozilla/5.0'})
print(f'Direct download: {r2.status_code}, headers: {dict(r2.headers)}')
if r2.status_code == 200:
    print(f'Content length: {len(r2.content)}')
    # Save first 1KB to check
    print(r2.content[:200])
