import requests
import json

API_HOST = 'http://127.22.77.172:8080/local_control/goto'
headers = {}
data = {
 "goals":
    [
        {
            "locationX" : -10.36,
            "locationY" : -3.52,
            "orientation" : 0.23
        },
        {
            "locationX" : -9.22,
            "locationY" : -3.04,
            "orientation" : 0.201
        }
    ]   
}

def req(path, method, data={}):
    url = API_HOST
    print('HTTP Method: %s' % method)
    print('Request URL: %s' % url)
    print('Headers: %s' % headers)

    if method == 'GET':
        return requests.get(url, headers=headers)
    else:
        return requests.post(url, headers=headers, data=json.dumps(data))

resp = req('', 'post', data)
print("response status:\n%d" % resp.status_code)
print("response headers:\n%s" % resp.headers)
print("response body:\n%s" % resp.text)
