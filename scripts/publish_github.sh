#!/bin/bash
# 把本仓库发布到 GitHub：创建仓库（已存在则复用）+ 推送当前分支 + 写好简介和话题。
#
#   GH_TOKEN=ghp_xxx bash scripts/publish_github.sh [仓库名] [public|private]
#
# 令牌用 GitHub 的 classic PAT，勾上 repo 权限即可（要放到已有组织的仓库里时另说）。
# 脚本不会把令牌写进 .git/config 或任何文件：只通过环境变量传给 git 的临时凭据助手。
# 用完请到 https://github.com/settings/tokens 把令牌撤销。
set -euo pipefail

REPO="${1:-K100LC-RT4}"
VIS="${2:-public}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESC="海光 K100 LC（gfx926）上从零自研的 Qwen3.8-27B 推理运行时：RT4 权重格式、int4 算子、MTP3 投机解码、本地视觉塔与 OpenAI 兼容服务"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "需要 GitHub 令牌：GH_TOKEN=ghp_xxx bash scripts/publish_github.sh" >&2
  echo "到 https://github.com/settings/tokens/new 建一个 classic token（勾 repo），复制以 ghp_ 开头的那串。" >&2
  exit 1
fi

api() { curl -sS -H "Authorization: token $GH_TOKEN" -H 'Accept: application/vnd.github+json' "$@"; }

LOGIN="$(api https://api.github.com/user | python3 -c 'import json,sys; print(json.load(sys.stdin)["login"])')"
echo "账号：$LOGIN"

# 仓库不存在（404）才创建；已存在就直接推
HTTP="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: token $GH_TOKEN" \
        "https://api.github.com/repos/$LOGIN/$REPO")"
if [ "$HTTP" = "404" ]; then
  private=false; [ "$VIS" = "private" ] && private=true
  api -X POST https://api.github.com/user/repos \
    -d "{\"name\":\"$REPO\",\"description\":\"$DESC\",\"private\":$private,\"has_issues\":true,\"has_wiki\":false}" \
    >/dev/null
  echo "已创建仓库 $LOGIN/$REPO（$VIS）"
else
  echo "仓库已存在（HTTP $HTTP），直接推送"
fi

api -X PATCH "https://api.github.com/repos/$LOGIN/$REPO" \
  -d '{"topics":["hip","rocm","llm-inference","quantization","int4","mtp","speculative-decoding","dcu","qwen"]}' \
  >/dev/null 2>&1 || true

cd "$ROOT"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$LOGIN/$REPO.git"
# 令牌只在这个 git 进程的临时凭据助手里用一次，不落盘
GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
  -c credential.helper='!f() { echo username=x-access-token; echo password="$GH_TOKEN"; }; f' \
  push -u origin HEAD:main

echo
echo "仓库地址：https://github.com/$LOGIN/$REPO"
