#!/bin/bash
#
# Download datasets for Spatial Adapter experiments.
#
# Prerequisites:
#   pip install kaggle
#   Place your Kaggle API key at ~/.kaggle/kaggle.json
#
# Usage:
#   bash data/download_data.sh
#

set -e
DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Data directory: $DATA_DIR"

# 1. KAUST 2b_8
echo ""
echo "=== [1/3] KAUST Competition 2b_8 ==="
KAUST_DIR="$DATA_DIR/kaust"
if [ -f "$KAUST_DIR/2b_8.csv" ]; then
    echo "Already exists: $KAUST_DIR/2b_8.csv — skipping."
else
    mkdir -p "$KAUST_DIR"
    echo "Downloading from Kaggle..."
    kaggle competitions download -c 2022-kaust-ss-competition-2b -p "$KAUST_DIR"
    echo "Extracting..."
    unzip -o "$KAUST_DIR/2022-kaust-ss-competition-2b.zip" -d "$KAUST_DIR"
    rm -f "$KAUST_DIR/2022-kaust-ss-competition-2b.zip"
    echo "Done: $KAUST_DIR/"
    ls -lh "$KAUST_DIR/"
fi

# 2. Weather2K
echo ""
echo "=== [2/3] Weather2K ==="
W2K_DIR="$DATA_DIR/weather2k"
if [ -f "$W2K_DIR/weather2k.npy" ]; then
    echo "Already exists: $W2K_DIR/weather2k.npy — skipping."
else
    mkdir -p "$W2K_DIR"
    echo "Weather2K .npy is not publicly downloadable via script."
    echo "Please copy your weather2k.npy to: $W2K_DIR/weather2k.npy"
    echo ""
    echo "If you have it elsewhere, run:"
    echo "  cp /path/to/weather2k.npy $W2K_DIR/"
fi

# 3. GWHD (Global Wheat Head Detection)
echo ""
echo "=== [3/3] Global Wheat Head Detection ==="
GWHD_DIR="$DATA_DIR/gwhd"
if [ -f "$GWHD_DIR/train.csv" ]; then
    echo "Already exists: $GWHD_DIR/train.csv — skipping."
else
    mkdir -p "$GWHD_DIR"
    echo "Downloading from Kaggle..."
    kaggle competitions download -c global-wheat-detection -p "$GWHD_DIR"
    echo "Extracting..."
    unzip -o "$GWHD_DIR/global-wheat-detection.zip" -d "$GWHD_DIR"
    rm -f "$GWHD_DIR/global-wheat-detection.zip"
    echo "Done: $GWHD_DIR/"
    ls -lh "$GWHD_DIR/" | head -10
fi

echo ""
echo "=== Summary ==="
echo "KAUST:    $([ -f "$KAUST_DIR/2b_8.csv" ] && echo 'OK' || echo 'MISSING')"
echo "Weather2K: $([ -f "$W2K_DIR/weather2k.npy" ] && echo 'OK' || echo 'MISSING — copy manually')"
echo "GWHD:     $([ -f "$GWHD_DIR/train.csv" ] && echo 'OK' || echo 'MISSING')"
