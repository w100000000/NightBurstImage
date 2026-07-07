#!/bin/sh
# Minimal capture pipeline for ELF2 Buildroot (no Python required)
# Usage: ./collect.sh <output_dir> <num_triplets>
# Example: ./collect.sh /data/train 100

OUTDIR="${1:-/tmp/nbinet_collect}"
COUNT="${2:-0}"  # 0 = continuous

SUBDEV=/dev/v4l-subdev2
VIDEO=/dev/video0

EXPOSURE_SHORT=3000
EXPOSURE_LONG=14300

echo "=== NBINet Collect (Shell) ==="
echo "Output: $OUTDIR"
echo "Target: $COUNT triplets"
echo ""

# Setup
echo "[Setup] Configuring sensor..."
v4l2-ctl -d $SUBDEV --set-subdev-fmt pad=0,width=4224,height=3136,code=0x3007
sleep 0.3
v4l2-ctl -d $SUBDEV -c vertical_blanking=11178
sleep 0.2

mkdir -p "$OUTDIR"

# Check for fuser
HAS_FUSER=0
which fuser >/dev/null 2>&1 && HAS_FUSER=1

kill_video() {
    if [ $HAS_FUSER -eq 1 ]; then
        PID=$(fuser /dev/video0 2>/dev/null)
        [ -n "$PID" ] && kill $PID 2>/dev/null
    fi
    sleep 0.1
}

i=0  # frame counter
t=0  # triplet counter
et="S"

echo "[Collect] Starting (Ctrl+C to stop)..."

while true; do
    # Set exposure
    if [ "$et" = "S" ]; then
        v4l2-ctl -d $SUBDEV -c exposure=$EXPOSURE_SHORT >/dev/null 2>&1
    else
        v4l2-ctl -d $SUBDEV -c exposure=$EXPOSURE_LONG >/dev/null 2>&1
    fi

    sleep 0.1

    # Kill stale v4l2 processes
    kill_video

    # Capture
    v4l2-ctl -d $VIDEO --stream-mmap --stream-count=1 --stream-to=/tmp/cap.raw >/dev/null 2>&1

    sz=$(wc -c < /tmp/cap.raw 2>/dev/null)
    if [ "$sz" -lt 1000000 ]; then
        echo "  Frame $i capture failed (size=$sz), retry"
        continue
    fi

    # Feed to ring buffer
    if [ "$et" = "S" ]; then
        cp /tmp/cap.raw /tmp/ring_s1.raw
        # Check if we have S-L-S
        if [ -f /tmp/ring_prev_s.raw ] && [ -f /tmp/ring_l.raw ]; then
            t=$((t + 1))
            DIR="${OUTDIR}/scene_$(printf '%05d' $t)"
            mkdir -p "$DIR"
            cp /tmp/ring_prev_s.raw "$DIR/short1.raw"
            cp /tmp/ring_l.raw       "$DIR/long.raw"
            cp /tmp/ring_s1.raw      "$DIR/short2.raw"
            echo "  Triplet $t → $DIR/"
        fi
        cp /tmp/cap.raw /tmp/ring_prev_s.raw
    else
        cp /tmp/cap.raw /tmp/ring_l.raw
    fi

    i=$((i + 1))
    echo "  Frame $i [$et]: ${sz} bytes"

    # Flip exposure
    if [ "$et" = "S" ]; then et="L"; else et="S"; fi

    # Check target
    if [ "$COUNT" -gt 0 ] && [ "$t" -ge "$COUNT" ]; then
        break
    fi
done

rm -f /tmp/ring_*.raw
echo "[Done] $t triplets saved"
