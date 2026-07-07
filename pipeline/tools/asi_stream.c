/*
 * asi_stream.c — 持续流压力测试: 连续抓 N 帧, 统计 reset / timeout
 *
 * 用途: 判断 RK3588 xHCI 在持续 SuperSpeed 流下是否反复 reset 崩溃
 * 只做纯采集: 不切换曝光, 不做NPU/显示/识别
 *
 * 用法: ./asi_stream [帧数] [曝光us] [带宽overload] [ROI宽] [ROI高] [bin]
 *   例: ./asi_stream 100 16667 -1 960 544 2   <- 只输出中心960x544, bin2
 *   ROI宽/高传0或不传 = 满幅(MaxW/bin × MaxH/bin); 宽自动对齐到8的倍数, 高对齐到2
 *   带宽传 -1 或不传 = 用SDK自动值
 * 编译: 参考 pipeline/core/Makefile (同 asi_snap)
 */
#include "ASICamera2.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

static long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int main(int argc, char *argv[])
{
    int  nframes = (argc > 1) ? atoi(argv[1]) : 200;
    int  expo_us = (argc > 2) ? atoi(argv[2]) : 16667;
    int  bw_set  = (argc > 3) ? atoi(argv[3]) : -1;  /* -1 = 不改, 用SDK自动值 */

    int num = ASIGetNumOfConnectedCameras();
    printf("cameras: %d\n", num);
    if (num <= 0) { printf("FAIL: no camera\n"); return 1; }

    ASI_ERROR_CODE r;
    r = ASIOpenCamera(0);   printf("open: %d\n", r);   if (r) return 1;
    r = ASIInitCamera(0);   printf("init: %d\n", r);   if (r) { ASICloseCamera(0); return 1; }

    ASI_CAMERA_INFO info;
    ASIGetCameraProperty(&info, 0);
    printf("name=%s max=%dx%d usb3=%d\n",
           info.Name, (int)info.MaxWidth, (int)info.MaxHeight, (int)info.IsUSB3Camera);

    int bin   = (argc > 6) ? atoi(argv[6]) : 2;
    int fullw = info.MaxWidth / bin, fullh = info.MaxHeight / bin;
    int w = (argc > 4 && atoi(argv[4]) > 0) ? atoi(argv[4]) : fullw;
    int h = (argc > 5 && atoi(argv[5]) > 0) ? atoi(argv[5]) : fullh;
    w &= ~7;   /* 宽对齐8的倍数 (ASI要求) */
    h &= ~1;   /* 高对齐2的倍数 */

    r = ASISetROIFormat(0, w, h, bin, ASI_IMG_RAW16);
    printf("ROI %dx%d bin=%d: %d %s\n", w, h, bin, r, r ? "<-- FAIL" : "");
    if (r) { ASICloseCamera(0); return 1; }

    /* 居中 ROI (binned 坐标系, 相对满幅居中; 起点对齐偶数保持Bayer相位) */
    if (w < fullw || h < fullh) {
        int sx = ((fullw - w) / 2) & ~1;
        int sy = ((fullh - h) / 2) & ~1;
        ASI_ERROR_CODE rs = ASISetStartPos(0, sx, sy);
        printf("StartPos (%d,%d): %d %s\n", sx, sy, rs, rs ? "<-- FAIL" : "");
    }
    printf("每帧数据量: %.2f MB\n", (double)w * h * 2 / (1024*1024));

    ASISetControlValue(0, ASI_EXPOSURE, expo_us, ASI_FALSE);
    ASISetControlValue(0, ASI_GAIN, 300, ASI_FALSE);

    /* 查询/设置 USB 带宽占用 (BANDWIDTHOVERLOAD) — 帧率关键 */
    long bw_cur = 0; ASI_BOOL bw_auto = ASI_FALSE;
    ASIGetControlValue(0, ASI_BANDWIDTHOVERLOAD, &bw_cur, &bw_auto);
    printf("bandwidth (auto): 当前=%ld auto=%d\n", bw_cur, bw_auto);
    if (bw_set >= 0) {
        ASISetControlValue(0, ASI_BANDWIDTHOVERLOAD, bw_set, ASI_FALSE);
        ASIGetControlValue(0, ASI_BANDWIDTHOVERLOAD, &bw_cur, &bw_auto);
        printf("bandwidth 已设为: %ld\n", bw_cur);
    }

    r = ASIStartVideoCapture(0);
    printf("start: %d\n", r);
    if (r) { ASICloseCamera(0); return 1; }

    size_t buf_sz = (size_t)w * h * 2;
    unsigned char *buf = (unsigned char *)malloc(buf_sz);
    int to_ms = expo_us / 1000 * 2 + 2000;

    printf("streaming %d frames @ %dus (timeout=%dms/frame)...\n",
           nframes, expo_us, to_ms);
    printf("frame  result  elapsed_ms\n");

    int ok = 0, fail = 0;
    long t0 = now_ms(), tprev = t0;

    for (int i = 0; i < nframes; i++) {
        r = ASIGetVideoData(0, buf, (long)buf_sz, to_ms);
        long tn = now_ms();
        if (r == ASI_SUCCESS) {
            ok++;
        } else {
            fail++;
            /* 每次失败都打印 (成功的每20帧打一次省屏幕) */
            printf("%5d  FAIL=%-2d  %ldms  <-- err\n", i, r, tn - tprev);
        }
        if (r == ASI_SUCCESS && (i % 20 == 0)) {
            printf("%5d  ok      %ldms\n", i, tn - tprev);
        }
        tprev = tn;
    }

    long total = now_ms() - t0;
    printf("\n=== 结果 ===\n");
    printf("总帧数:   %d\n", nframes);
    printf("成功:     %d\n", ok);
    printf("失败:     %d\n", fail);
    printf("总耗时:   %ldms\n", total);
    printf("平均fps:  %.1f\n", ok * 1000.0 / (total > 0 ? total : 1));
    printf("(同时另开终端跑 `dmesg -w | grep -i reset` 数 reset 次数)\n");

    ASIStopVideoCapture(0);
    ASICloseCamera(0);
    free(buf);
    printf("done.\n");
    return 0;
}
