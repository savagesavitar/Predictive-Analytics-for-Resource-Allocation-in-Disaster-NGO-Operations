import requests, json

resource_ids = [
    '9f1f8ea2-1',
    'd97707e6-9',
    '5bddf229-7',
    '9d4190cc-7',
]

base_url = 'https://ckandev.indiadataportal.com/dataset/25664936-d131-4b07-845e-4bd2c0d794aa/resource/'

for rid in resource_ids:
    patterns = [
        base_url + rid + '/download',
        base_url + rid + '/download/rainfall.csv',
        base_url + rid,
    ]
    for url in patterns:
        try:
            r = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
            ct = r.headers.get('Content-Type', '')
            if r.status_code == 200 and ('csv' in ct or 'text' in ct):
                fname = 'B:\\Predictive Analytics for Resource Allocation\\data\\raw\\rainfall_' + rid[:8] + '.csv'
                with open(fname, 'wb') as f:
                    f.write(r.content)
                print('OK:', url, '->', len(r.content), 'bytes')
                break
            elif r.status_code == 200:
                info = 'type=' + ct + ' len=' + str(len(r.content))
                print('Got 200 :', url, info)
        except Exception as e:
            print('ERR:', url, str(e))
    print('---')
