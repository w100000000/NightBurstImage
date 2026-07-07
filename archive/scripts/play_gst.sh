#!/bin/sh
# play_gst.sh — ELF2 视频播放 (使用 gstreamer waylandsink)
#
# 用法:
#   sh /mnt/sdcard/play_gst.sh                    # 默认: /mnt/sdcard/video.mp4
#   sh /mnt/sdcard/play_gst.sh /path/to/video.mp4  # 指定文件
#
# 依赖: gst-play-1.0 (ELF2 buildroot 自带)
# 说明:
#   - 使用 waylandsink 通过 Weston 显示到 MIPI 屏幕
#   - 不需要停止 Weston (官方标准方案)
#   - mppvideodec 硬件解码 H.264/H.265
#
# 如果 gst-play-1.0 不可用, 回退到 gst-launch-1.0 pipeline

VIDEO="${1:-/mnt/sdcard/video.mp4}"

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

echo "=== ELF2 Video Player ==="
echo "Video: $VIDEO"
echo "Display: waylandsink (Weston)"
echo "========================="

# 检查 Weston 是否在运行
if ! pidof weston > /dev/null 2>&1; then
    echo "Weston not running, starting..."
    /etc/init.d/S49weston start 2>/dev/null
    sleep 2
fi

# 确认文件扩展名
EXT="${VIDEO##*.}"

# 优先使用 gst-play-1.0 (自动选择解码器+sink, 最简单)
if command -v gst-play-1.0 > /dev/null 2>&1; then
    echo "Using gst-play-1.0..."
    gst-play-1.0 "$VIDEO"
    echo "Playback finished."
    exit 0
fi

# 回退: 用 gst-launch-1.0
if command -v gst-launch-1.0 > /dev/null 2>&1; then
    echo "gst-play-1.0 not found, using gst-launch-1.0..."

    # 根据文件类型构建 pipeline
    case "$EXT" in
        mp4|MP4)
            # MP4 container - H.264/H.265
            echo "Pipeline: filesrc → qtdemux → parse → mppvideodec → waylandsink"
            gst-launch-1.0 filesrc location="$VIDEO" ! \
                qtdemux ! queue ! h264parse ! mppvideodec ! waylandsink
            ;;
        mkv|MKV|webm|WEBM)
            echo "Pipeline: filesrc → matroskademux → mppvideodec → waylandsink"
            gst-launch-1.0 filesrc location="$VIDEO" ! \
                matroskademux ! queue ! mppvideodec ! waylandsink
            ;;
        *)
            # 通用
            echo "Pipeline: filesrc → decodebin → mppvideodec → waylandsink (auto)"
            gst-launch-1.0 filesrc location="$VIDEO" ! \
                decodebin ! mppvideodec ! waylandsink
            ;;
    esac

    echo "Playback finished."
    exit 0
fi

echo "ERROR: Neither gst-play-1.0 nor gst-launch-1.0 found!"
exit 1
