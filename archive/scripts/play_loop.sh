#!/bin/sh
# play_loop.sh — loop video playback for promo filming
# Usage: sh /mnt/sdcard/play_loop.sh [video.mp4]

VIDEO="${1:-/mnt/sdcard/video.mp4}"
LOOP_DELAY="${2:-1}"

# Wayland env (not set in SSH or init scripts)
export XDG_RUNTIME_DIR=/run
export WAYLAND_DISPLAY=wayland-0

echo "=== ELF2 Loop Player ==="
echo "Video: $VIDEO"
echo "Mode:  infinite loop"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "========================"

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    ls -la /mnt/sdcard/*.mp4 2>/dev/null || echo "  (no mp4 files)"
    exit 1
fi

# Wait for Weston
echo "Waiting for Weston..."
for i in $(seq 1 15); do
    if pidof weston > /dev/null 2>&1; then
        echo "Weston running (PID $(pidof weston))"
        break
    fi
    sleep 1
done

LOOP=0
while true; do
    LOOP=$((LOOP + 1))
    echo ""
    echo "=== Loop $LOOP ==="

    if command -v gst-play-1.0 > /dev/null 2>&1; then
        gst-play-1.0 "$VIDEO"
    elif command -v gst-launch-1.0 > /dev/null 2>&1; then
        gst-launch-1.0 filesrc location="$VIDEO" ! \
            qtdemux ! queue ! h264parse ! mppvideodec ! waylandsink
    else
        echo "ERROR: No gstreamer tools found!"
        exit 1
    fi

    echo "Loop $LOOP done. Restarting in ${LOOP_DELAY}s..."
    sleep $LOOP_DELAY
done
