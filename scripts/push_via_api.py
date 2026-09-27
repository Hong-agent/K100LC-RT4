#!/usr/bin/env python3
"""本机 git-over-HTTPS 到 github.com 不通/不稳时，用 GitHub Git Data API 推送。

和 `git push` 的区别：**按内容比对**——把本地 `HEAD` 的每个文件 blob 与远端 tree 比对，
只在 sha 不同时上传，所以本地提交历史与远端分叉也能照推（代价是每个改动文件一个 REST
请求，且只适合文本仓库，不适合塞大二进制）。

    GH_TOKEN=ghp_xxx python3 scripts/push_via_api.py --dry-run    # 先看会改什么
    GH_TOKEN=ghp_xxx python3 scripts/push_via_api.py              # 真推
    GH_TOKEN=ghp_xxx python3 scripts/push_via_api.py --branch main

令牌只从环境变量读，不落盘；用完请到 GitHub 设置里撤销。
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args, binary=False):
    out = subprocess.check_output(['git', *args], cwd=ROOT)
    return out if binary else out.decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', default='main')
    ap.add_argument('--repo', default='', help='owner/repo（默认从 origin 解析）')
    ap.add_argument('--message', default='', help='提交信息（默认取本地 HEAD 的）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    token = os.environ.get('GH_TOKEN', '')
    if not token:
        sys.exit('需要 GH_TOKEN=ghp_xxx（classic PAT，勾 repo 权限）')
    repo = args.repo
    if not repo:
        url = git('remote', 'get-url', 'origin').strip()
        m = re.search(r'github\.com[:/]([^/]+/[^/.]+)', url)
        if not m:
            sys.exit(f'从 origin 解析不出 owner/repo：{url}（用 --repo 指定）')
        repo = m.group(1)

    def api(method, path, body=None, tries=6):
        """所有调用带退避重试：这台机器到 GitHub 的连接会间歇性断。"""
        last = None
        for i in range(tries):
            req = urllib.request.Request(
                'https://api.github.com' + path, method=method,
                data=None if body is None else json.dumps(body).encode(),
                headers={'Authorization': 'token ' + token,
                         'Accept': 'application/vnd.github+json',
                         'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    text = r.read().decode()
                    return json.loads(text) if text else {}
            except urllib.error.HTTPError as e:
                text = e.read().decode()[:200]
                if e.code < 500 and e.code != 429:
                    sys.exit(f'HTTP {e.code} {method} {path}: {text}')
                last = f'HTTP {e.code} {text}'
            except Exception as e:                                   # noqa: BLE001
                last = repr(e)[:120]
            wait = min(30, 2 ** i)
            print(f'  重试 {i + 1}/{tries}：{last}（等 {wait}s）', file=sys.stderr)
            time.sleep(wait)
        sys.exit(f'放弃：{last}')

    head = git('rev-parse', 'HEAD').strip()
    parent = git('rev-parse', 'HEAD~1').strip() if git('rev-list', '--count', 'HEAD').strip() != '1' else None
    message = args.message or git('log', '-1', '--format=%B').strip()
    an, ae = git('log', '-1', '--format=%an%n%ae').split()

    # 本地文件清单（path → blob sha / mode）
    local = {}
    for line in git('ls-tree', '-r', '-z', 'HEAD').split('\x00'):
        if not line:
            continue
        meta, path = line.split('\t', 1)
        mode, _type, sha = meta.split()
        local[path] = (mode, sha)

    remote_commit = api('GET', f'/repos/{repo}/commits/{args.branch}')
    remote_tree_sha = remote_commit['commit']['tree']['sha']
    remote_entries = api('GET', f'/repos/{repo}/git/trees/{remote_tree_sha}?recursive=1')['tree']
    remote = {e['path']: e['sha'] for e in remote_entries if e['type'] == 'blob'}

    changed = sorted(p for p in local if remote.get(p) != local[p][1])
    deleted = sorted(p for p in remote if p not in local)
    print(f'仓库 {repo}@{args.branch}（远端 {remote_commit["sha"][:9]}，本地 {head[:9]}）')
    print(f'改动 {len(changed)} 个文件，删除 {len(deleted)} 个')
    for p in changed:
        print('  改', p)
    for p in deleted:
        print('  删', p)
    if not changed and not deleted:
        print('内容已一致，无需推送')
        return
    if args.dry_run:
        print('（--dry-run：到此为止）')
        return

    entries = []
    for p in changed:
        mode, _sha = local[p]
        with open(os.path.join(ROOT, p), 'rb') as f:
            blob = api('POST', f'/repos/{repo}/git/blobs',
                       {'content': base64.b64encode(f.read()).decode(), 'encoding': 'base64'})
        entries.append({'path': p, 'mode': mode, 'type': 'blob', 'sha': blob['sha']})
        print(f'  上传 {p}')
    entries += [{'path': p, 'mode': '100644', 'type': 'blob', 'sha': None} for p in deleted]

    tree = api('POST', f'/repos/{repo}/git/trees',
               {'base_tree': remote_tree_sha, 'tree': entries})
    commit = api('POST', f'/repos/{repo}/git/commits',
                 {'message': message, 'tree': tree['sha'],
                  'parents': [remote_commit['sha']],
                  'author': {'name': an, 'email': ae},
                  'committer': {'name': an, 'email': ae}})
    api('PATCH', f'/repos/{repo}/git/refs/heads/{args.branch}',
        {'sha': commit['sha'], 'force': True})
    print(f'已推送：https://github.com/{repo}/commit/{commit["sha"]}')
    print('注意：远端这次提交的 sha 与本地不同（GitHub 会归一化元数据），但树内容一致。')


if __name__ == '__main__':
    main()
