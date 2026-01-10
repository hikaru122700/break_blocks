#!/bin/bash

# 自動プッシュスクリプト
# 10秒ごとに変更をチェックし、変更があれば自動でcommit & push

REPO_DIR="/mnt/c/Users/owner/claude_code/break_blocks"
INTERVAL=10

cd "$REPO_DIR" || exit 1

echo "==================================="
echo "自動プッシュを開始します"
echo "対象: $REPO_DIR"
echo "間隔: ${INTERVAL}秒"
echo "停止: Ctrl+C"
echo "==================================="

while true; do
    # 変更があるかチェック
    if [[ -n $(git status --porcelain) ]]; then
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

        # 変更をステージング
        git add -A

        # コミット
        git commit -m "auto-commit: $TIMESTAMP"

        # プッシュ
        if git push; then
            echo "[$TIMESTAMP] プッシュ完了"
        else
            echo "[$TIMESTAMP] プッシュ失敗"
        fi
    else
        echo "[$(date '+%H:%M:%S')] 変更なし"
    fi

    sleep $INTERVAL
done
