#!/bin/sh
# 屏幕重摄采集脚本 — 暗室 + 显示屏 + OV13855
# 用法: ./collect_screen.sh <output_dir> <scene_count>
#
# 每场景操作:
#   1. 屏幕显示 S1.png → 按 Enter → 拍短曝光 RAW
#   2. 屏幕显示 L.png  → 按 Enter → 拍长曝光 RAW
#   3. 屏幕显示 S2.png → 按 Enter → 拍短曝光 RAW
#   4. 屏幕显示 GT.png → 按 Enter → 拍 50 帧长曝光平均 RAW
#
# 曝光: 短=3000 lines (~16ms), 长=14300 lines (~82ms)

OUTDIR="${1:-/data/nbinet_ov13855}"
SCENES="${2:-10}"

SUBDEV=/dev/v4l-subdev2
VIDEO=/dev/video0
TMP=/tmp/screen_cap.raw

EXPOSURE_SHORT=3000
EXPOSURE_LONG=14300
GT_FRAMES=50  # GT 平均帧数

echo "============================================"
echo "  Screen Re-photography Capture"
echo "  Output: $OUTDIR"
echo "  Scenes: $SCENES"
echo "============================================"
echo ""

# --- 初始化传感器 ---
setup_sensor() {
    echo "[Init] Configuring sensor..."
    v4l2-ctl -d $SUBDEV --set-subdev-fmt pad=0,width=4224,height=3136,code=0x3007
    sleep 0.3
    v4l2-ctl -d $SUBDEV -c vertical_blanking=11178
    sleep 0.2
    echo "[Init] Done."
}

# --- 杀残留 ---
kill_video() {
    PID=$(fuser /dev/video0 2>/dev/null)
    [ -n "$PID" ] && kill $PID 2>/dev/null
    sleep 0.05
}

# --- 拍一帧 ---
capture_one() {
    local exp=$1
    local out=$2
    v4l2-ctl -d $SUBDEV -c exposure=$exp >/dev/null 2>&1
    sleep 0.15
    kill_video
    v4l2-ctl -d $VIDEO --stream-mmap --stream-count=1 --stream-to=$out >/dev/null 2>&1
    local sz=$(wc -c < $out 2>/dev/null)
    if [ "$sz" -lt 1000000 ]; then
        return 1
    fi
    return 0
}

# --- 拍 GT: 多帧平均 (纯 shell, 用 dd 叠加 16bit, 最后除帧数) ---
capture_gt_average() {
    local scene_dir=$1
    local tmpdir=/tmp/gt_frames
    rm -rf $tmpdir
    mkdir -p $tmpdir

    echo "    Capturing $GT_FRAMES long-exposure frames for GT..."
    for f in $(seq 1 $GT_FRAMES); do
        capture_one $EXPOSURE_LONG "$tmpdir/frame_$f.raw" || {
            echo "    Frame $f failed, retrying..."
            sleep 0.1
            capture_one $EXPOSURE_LONG "$tmpdir/frame_$f.raw"
        }
        printf "    %d/%d\r" $f $GT_FRAMES
    done
    echo ""

    # 纯 shell 浮点平均: 用 Python 做 (Buildroot 可能有 micropython)
    # fallback: 保存全部帧, 训练时再平均
    echo "    Averaging frames..."
    python3 -c "
import struct, sys
n = $GT_FRAMES
# Read first frame to get size
with open('$tmpdir/frame_1.raw', 'rb') as f:
    data = f.read()
total_bytes = len(data)
# Accumulate as uint16
acc = [0] * (total_bytes // 2)
for i in range(1, n + 1):
    with open(f'$tmpdir/frame_{i}.raw', 'rb') as f:
        raw = f.read()
    for j in range(0, total_bytes, 2):
        acc[j//2] += raw[j] | (raw[j+1] << 8)
# Average and write
with open('$scene_dir/GT.raw', 'wb') as f:
    for v in acc:
        avg = v // n
        f.write(struct.pack('<H', avg))
print('    GT saved: ${scene_dir}/GT.raw')
" 2>/dev/null || {
        # Python3 not available — just save all frames, use first as proxy
        echo "    Python3 not available, saving raw frames instead"
        mkdir -p "$scene_dir/gt_frames"
        cp $tmpdir/frame_*.raw "$scene_dir/gt_frames/"
        # Use frame 1 as placeholder GT
        cp "$tmpdir/frame_1.raw" "$scene_dir/GT.raw"
        echo "    GT frames saved to $scene_dir/gt_frames/ (process offline)"
    }
    rm -rf $tmpdir
}

# --- 主流程 ---
setup_sensor

scene=1
while [ $scene -le $SCENES ]; do
    DIR="$OUTDIR/scene_$(printf '%05d' $scene)"
    mkdir -p "$DIR"

    echo ""
    echo "=== Scene $scene/$SCENES ==="
    echo "    Dir: $DIR"

    # 1) Short frame 1
    echo -n "  [1/4] Display S1.png on screen, then press Enter..."
    read _dummy
    if capture_one $EXPOSURE_SHORT $TMP; then
        cp $TMP "$DIR/short1.raw"
        echo " OK ($(wc -c < $DIR/short1.raw) bytes)"
    else
        echo " FAIL"
    fi

    # 2) Long frame
    echo -n "  [2/4] Display L.png on screen, then press Enter..."
    read _dummy
    if capture_one $EXPOSURE_LONG $TMP; then
        cp $TMP "$DIR/long.raw"
        echo " OK ($(wc -c < $DIR/long.raw) bytes)"
    else
        echo " FAIL"
    fi

    # 3) Short frame 2
    echo -n "  [3/4] Display S2.png on screen, then press Enter..."
    read _dummy
    if capture_one $EXPOSURE_SHORT $TMP; then
        cp $TMP "$DIR/short2.raw"
        echo " OK ($(wc -c < $DIR/short2.raw) bytes)"
    else
        echo " FAIL"
    fi

    # 4) GT — 多帧长曝光平均
    echo -n "  [4/4] Display GT.png on screen, then press Enter..."
    read _dummy
    capture_gt_average "$DIR"

    scene=$((scene + 1))
done

echo ""
echo "=== Done: $SCENES scenes saved to $OUTDIR/ ==="
