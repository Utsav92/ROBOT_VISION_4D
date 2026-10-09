"""Run a Python file (or inline code) inside the live TouchDesigner via the MCP web server.
usage:  python tdrun.py file.py            -> executes file contents in TD
        python tdrun.py -c "code"          -> executes inline
Script may set `result = ...`; stdout/stderr of the script are echoed.
"""
import sys, json, urllib.request

URL = 'http://127.0.0.1:9981/api/td/server/exec'


def run(code, timeout=300):
    req = urllib.request.Request(URL, data=json.dumps({'script': code}).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


if __name__ == '__main__':
    if sys.argv[1] == '-c':
        code = sys.argv[2]
    else:
        code = open(sys.argv[1], encoding='utf-8').read()
    try:
        out = run(code)
    except Exception as e:
        print('HTTP ERROR', e); sys.exit(2)
    d = out.get('data') or {}
    if d.get('stdout'): print(d['stdout'], end='')
    if d.get('stderr'): print('STDERR:', d['stderr'], end='')
    if out.get('error'): print('ERROR:', out['error'])
    if 'result' in d and d['result'] is not None:
        print('RESULT:', json.dumps(d['result'], indent=1, default=str)[:60000])

