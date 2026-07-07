/*
 * asi_snap.c — 极简测试: 拍一帧 RAW16 存盘
 *
 * 用法: ./asi_snap [输出文件名]
 * 编译: 参考 pipeline/core/Makefile
 */
#include "ASICamera2.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    const char *outfile = (argc > 1) ? argv[1] : "/tmp/asi_snap.raw";

    /* 1. 检测相机 */
    int num = ASIGetNumOfConnectedCameras();
    printf("cameras: %d\n", num);
    if (num <= 0) { printf("FAIL: no camera\n"); return 1; }

    /* 2. 打开 */
    ASI_ERROR_CODE r;
    r = ASIOpenCamera(0);    printf("open: %d\n", r);
    if (r) return 1;

    /* 3. 初始化 */
    r = ASIInitCamera(0);    printf("init: %d\n", r);
    if (r) { ASICloseCamera(0); return 1; }

    /* 4. 相机信息 */
    ASI_CAMERA_INFO info;
    ASIGetCameraProperty(&info, 0);
    printf("name=%s max=%dx%d bayer=%d bit=%d usb3=%d\n",
           info.Name, info.MaxWidth, info.MaxHeight,
           info.BayerPattern, info.BitDepth, info.IsUSB3Camera);

    /* 5. 设置 ROI (2x2 binning → 1920x1080, 降低数据量) */
    int w = info.MaxWidth / 2;
    int h = info.MaxHeight / 2;
    r = ASISetROIFormat(0, w, h, 2, ASI_IMG_RAW16);
    printf("ROI %dx%d bin=2: %d\n", w, h, r);
    if (r) { ASICloseCamera(0); return 1; }

    /* 6. 设置曝光 (固定, 不切换) */
    ASISetControlValue(0, ASI_EXPOSURE, 16667, ASI_FALSE);
    ASISetControlValue(0, ASI_GAIN, 300, ASI_FALSE);

    /* 7. 开始采集 */
    r = ASIStartVideoCapture(0);
    printf("start: %d\n", r);
    if (r) { ASICloseCamera(0); return 1; }

    printf("ready: %dx%d, waiting for 1st frame...\n", w, h);

    /* 8. 丢掉前 2 帧 (稳定曝光) */
    size_t buf_sz = (size_t)w * h * 2;
    unsigned char *buf = (unsigned char *)malloc(buf_sz);
    for (int d = 0; d < 2; d++) {
        r = ASIGetVideoData(0, buf, (long)buf_sz, 5000);
        printf("discard %d: %d\n", d, r);
        if (r) { printf("FAIL: discard frame err=%d\n", r); free(buf); ASIStopVideoCapture(0); ASICloseCamera(0); return 1; }
    }

    /* 9. 拿一帧有效数据 */
    r = ASIGetVideoData(0, buf, (long)buf_sz, 5000);
    printf("frame: %d\n", r);
    if (r) { printf("FAIL: get frame err=%d\n", r); }

    /* 10. 存盘 */
    FILE *f = fopen(outfile, "wb");
    if (f) {
        size_t n = fwrite(buf, 1, buf_sz, f);
        fclose(f);
        printf("saved %zu bytes to %s\n", n, outfile);
    } else {
        printf("FAIL: can't write %s\n", outfile);
    }

    /* 11. 清理 */
    ASIStopVideoCapture(0);
    ASICloseCamera(0);
    free(buf);
    printf("done.\n");
    return 0;
}
