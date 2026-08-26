#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="llm-security-cli"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
ROUTER_FILE="$SCRIPT_DIR/artifacts/phase2e/router_anchor_rare_v2.pkl"
OUTPUT_DIR="$SCRIPT_DIR/output"

# 1. 폴더 경로를 입력했는지 확인
if [ "$#" -ne 1 ]; then
    echo "사용법: ./run.sh <분석할 C/C++ 폴더>"
    exit 1
fi

# 2. 입력 폴더가 실제로 존재하는지 확인
if [ ! -d "$1" ]; then
    echo "오류: 폴더를 찾을 수 없습니다: $1"
    exit 1
fi

INPUT_DIR="$(cd "$1" && pwd)"

# 3. Docker 확인
if ! command -v docker >/dev/null 2>&1; then
    echo "오류: Docker가 설치되어 있지 않습니다."
    exit 1
fi

# 4. .env 확인
if [ ! -f "$ENV_FILE" ]; then
    echo "오류: .env 파일을 찾을 수 없습니다."
    exit 1
fi

# 5. Router 모델 확인
if [ ! -f "$ROUTER_FILE" ]; then
    echo "오류: Router artifact를 찾을 수 없습니다."
    exit 1
fi

# 6. 결과 폴더 생성
mkdir -p "$OUTPUT_DIR"

# 7. Docker 이미지가 없으면 자동으로 빌드
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[1/3] Docker 이미지가 없어 새로 빌드합니다."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.cli" \
        -t "$IMAGE_NAME" \
        "$SCRIPT_DIR"
else
    echo "[1/3] Docker 이미지 확인 완료"
fi

echo "[2/3] 취약점 분석 시작"
echo "분석 대상: $INPUT_DIR"

# 8. Docker에서 분석 실행
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$INPUT_DIR:/input:ro" \
    -v "$ENV_FILE:/app/.env:ro" \
    -v "$SCRIPT_DIR/artifacts:/app/artifacts:ro" \
    -v "$OUTPUT_DIR:/output" \
    "$IMAGE_NAME" \
    /input \
    --env-file /app/.env \
    --router-artifact /app/artifacts/phase2e/router_anchor_rare_v2.pkl \
    --output /output/analysis.json

echo "[3/3] 분석 완료"
echo "결과 파일: $OUTPUT_DIR/analysis.json"
