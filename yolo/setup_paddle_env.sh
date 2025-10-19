#!/bin/bash
# PaddlePaddle macOS 환경 설정 스크립트

echo "PaddlePaddle macOS 환경 설정..."

# OpenMP 충돌 방지
export KMP_DUPLICATE_LIB_OK=TRUE

# PaddlePaddle CPU 모드 강제
export FLAGS_use_mkldnn=False
export FLAGS_use_ngraph=False

# 로깅 레벨 설정
export GLOG_v=0
export GLOG_logtostderr=0

# YOLO와의 충돌 방지를 위한 스레드 설정
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "환경변수 설정 완료:"
echo "  KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK"
echo "  FLAGS_use_mkldnn=$FLAGS_use_mkldnn"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""
echo "사용법:"
echo "  source setup_paddle_env.sh"
echo "  python test_full_pipeline.py"
