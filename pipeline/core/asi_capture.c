/*
 * asi_capture.c — ZWO ASI585MC USB 相机采集 + 交替曝光
 *
 * ASI585MC = Sony IMX585 sensor (RGGB Bayer, 12-bit)
 * 与模型训练数据完全一致——无需色域转换
 *
 * 输出 raw_buf_t 到 raw_ring (每帧 4-plane float32, RGGB)
 */

#include "pipeline_types.h"
#include "ASICamera2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ASI_BLACK     96.0f    /* IMX585 黑电平 */
#define ASI_MAX_VAL   4095.0f  /* 12-bit 白电平 */
#define ASI_RANGE     (ASI_MAX_VAL - ASI_BLACK)  /* 3999 */
#define ASI_CAM_ID    0

static int asi_connected = 0;
static int asi_img_w = 0, asi_img_h = 0;

static int asi_init(pipeline_t *p)
{
    int num = ASIGetNumOfConnectedCameras();
    printf("[ASI] cameras detected: %d\n", num);
    if (num <= 0) return -1;

    ASI_ERROR_CODE ret;
    ret = ASIOpenCamera(ASI_CAM_ID);
    printf("[ASI] open: %d\n", ret);
    if (ret != ASI_SUCCESS) return -1;

    ret = ASIInitCamera(ASI_CAM_ID);
    printf("[ASI] init: %d\n", ret);
    if (ret != ASI_SUCCESS) { ASICloseCamera(ASI_CAM_ID); return -1; }

    /* 查询相机属性 */
    ASI_CAMERA_INFO info;
    ret = ASIGetCameraProperty(&info, 0);
    printf("[ASI] name=%s max=%dx%d pixel=%dum bayer=%d\n",
           info.Name, info.MaxWidth, info.MaxHeight, info.PixelSize, info.BayerPattern);
    printf("[ASI] isColor=%d isUSB3=%d bitDepth=%d\n",
           info.IsColorCam, info.IsUSB3Camera, info.BitDepth);

    /* 硬件 ROI: bin=2 下直接输出中心 BAYER_W×BAYER_H (960×544),
     * 正好是模型所需区域。相比满幅 1920×1080 传输量砍到 1/4,
     * USB2 帧率 2fps→8fps。StartPos 与原软件裁切偏移一致, 画面/Bayer相位不变。 */
    int bin   = 2;
    int fullw = info.MaxWidth  / bin;   /* 1920 */
    int fullh = info.MaxHeight / bin;   /* 1080 */
    asi_img_w = BAYER_W;                 /* 960 */
    asi_img_h = BAYER_H;                 /* 544 */
    ret = ASISetROIFormat(ASI_CAM_ID, asi_img_w, asi_img_h, bin, ASI_IMG_RAW16);
    printf("[ASI] bin=%d ROI %dx%d: %d\n", bin, asi_img_w, asi_img_h, ret);
    if (ret != ASI_SUCCESS) {
        ASICloseCamera(ASI_CAM_ID); return -1;
    }
    
    /* 居中 (binned 坐标系, 起点偶数保持 Bayer 相位) */
    int sx = ((fullw - asi_img_w) / 2) & ~1;   /* 480 */
    int sy = ((fullh - asi_img_h) / 2) & ~1;   /* 268 */
    ret = ASISetStartPos(ASI_CAM_ID, sx, sy);
    printf("[ASI] StartPos (%d,%d): %d\n", sx, sy, ret);

    ASISetControlValue(ASI_CAM_ID, ASI_EXPOSURE, p->expo_short_us, ASI_FALSE);
    ASISetControlValue(ASI_CAM_ID, ASI_GAIN, 0, ASI_FALSE);
    ASISetControlValue(ASI_CAM_ID, ASI_WB_R, 50, ASI_FALSE);
    ASISetControlValue(ASI_CAM_ID, ASI_WB_B, 50, ASI_FALSE);
    /* 关闭 HIGH_SPEED_MODE — 有些 USB 主控不兼容会触发 reset */
    /* ASISetControlValue(ASI_CAM_ID, ASI_HIGH_SPEED_MODE, 1, ASI_FALSE); */

    ret = ASIStartVideoCapture(ASI_CAM_ID);
    printf("[ASI] start capture: %d\n", ret);
    if (ret != ASI_SUCCESS) { ASICloseCamera(ASI_CAM_ID); return -1; }

    asi_connected = 1;
    printf("[ASI] ready: %dx%d RAW16 RGGB\n", asi_img_w, asi_img_h);
    return 0;
}

static void asi_set_exposure_gain(int expo_us, int gain) {
    ASISetControlValue(ASI_CAM_ID, ASI_EXPOSURE, expo_us, ASI_FALSE);
    ASISetControlValue(ASI_CAM_ID, ASI_GAIN, gain, ASI_FALSE);
}

static void asi_cleanup(void) {
    if (asi_connected) { ASIStopVideoCapture(ASI_CAM_ID); ASICloseCamera(ASI_CAM_ID); asi_connected = 0; }
}



static void bayer_extract(const uint16_t *bayer, int bw, int bh,
                          float *r, float *gr, float *gb, float *b,
                          int cx, int cy, int bayer_pat, float maxval)
{
    for (int y = 0; y < PLANE_H; y++) {
        int by = cy + 2*y, by1 = by+1;
        for (int x = 0; x < PLANE_W; x++) {
            int bx = cx + 2*x, bx1 = bx+1, idx = y*PLANE_W + x;
            uint16_t v00 = bayer[by *bw + bx];
            uint16_t v01 = bayer[by *bw + bx1];
            uint16_t v10 = bayer[by1*bw + bx];
            uint16_t v11 = bayer[by1*bw + bx1];
            /* 裁剪到 [黑电平, 最大值] 再归一化到 [0,1]，与训练一致 */
            if (v00 > 4095) v00 = 4095; if (v00 < 96) v00 = 96;
            if (v01 > 4095) v01 = 4095; if (v01 < 96) v01 = 96;
            if (v10 > 4095) v10 = 4095; if (v10 < 96) v10 = 96;
            if (v11 > 4095) v11 = 4095; if (v11 < 96) v11 = 96;
            float p00 = ((float)v00 - ASI_BLACK) / ASI_RANGE;
            float p01 = ((float)v01 - ASI_BLACK) / ASI_RANGE;
            float p10 = ((float)v10 - ASI_BLACK) / ASI_RANGE;
            float p11 = ((float)v11 - ASI_BLACK) / ASI_RANGE;
            switch (bayer_pat) {
            case 1: /* BGGR：B=00, Gb=01, Gr=10, R=11 */
                r[idx]=p11; gr[idx]=p10; gb[idx]=p01; b[idx]=p00; break;
            case 2: /* GRBG */
                r[idx]=p01; gr[idx]=p00; gb[idx]=p11; b[idx]=p10; break;
            case 3: /* GBRG */
                r[idx]=p10; gr[idx]=p11; gb[idx]=p00; b[idx]=p01; break;
            default: /* RGGB：R=00, Gr=01, Gb=10, B=11 */
                r[idx]=p00; gr[idx]=p01; gb[idx]=p10; b[idx]=p11; break;
            }
        }
    }
}

void *capture_thread(void *arg)
{
    pipeline_t *p = (pipeline_t *)arg;

    if (asi_init(p) < 0) { fprintf(stderr,"[Capture] ASI init fail\n"); return NULL; }

    /* 训练用 RGGB(=0) 提取 (dataset_imx585.py: R=左上 Gr=右上 Gb=左下 B=右下).
     * SDK 上报 BGGR(1) 是它自己的约定, 但模型输入必须按训练的 RGGB 提取.
     * 注意: SDK open 后二次 ASIGetCameraProperty 返回全0(bug), 故直接写死。 */
    int bayer_pat = 0;   /* RGGB — 匹配训练 imx585_bayer_to_rggb */
    float maxval = 4095.0f;
    printf("[Capture] Bayer=RGGB(%d) maxval=%.0f (匹配训练 dataset_imx585)\n",
           bayer_pat, maxval);

    /* 相机硬件 ROI 已是 960×544 (=BAYER_W×BAYER_H), cx=cy=0 不再裁切;
     * 若改回满幅采集, 此处会自动居中裁切, 逻辑通用。 */
    int cx = ((asi_img_w - BAYER_W) / 2) & ~1;
    int cy = ((asi_img_h - BAYER_H) / 2) & ~1;
    printf("[Capture] %dx%d → crop %dx%d at (%d,%d)\n",
           asi_img_w, asi_img_h, BAYER_W, BAYER_H, cx, cy);

    size_t buf_sz = (size_t)asi_img_w * asi_img_h * 2;
    unsigned char *asi_buf = (unsigned char *)malloc(buf_sz);

    int fid = 0;
    int expo_seq   = 0;  /* 0=short, 1=long, 2=short */
    int prev_expo  = -1; /* 上次曝光类型, 用于跳过冗余丢弃 */

    /* 匹配训练: S=16.7ms/gain=300, L=83.3ms/gain=0 */
    int s_exp  = (p->expo_short_us > 0) ? p->expo_short_us : 16667;
    int l_exp  = (p->expo_long_us  > 0) ? p->expo_long_us  : 83333;
    int s_gain = 300, l_gain = 0;

    /* 先设短曝光 + 丢弃 1 帧稳定 */
    asi_set_exposure_gain(s_exp, s_gain);
    printf("[Capture] S: %dus gain=%d  L: %dus gain=%d\n", s_exp, s_gain, l_exp, l_gain);
    int to_ms = s_exp/1000*2 + 2000;
    ASIGetVideoData(ASI_CAM_ID, asi_buf, (long)buf_sz, to_ms); /* 丢弃 1 帧 */
    prev_expo = EXP_SHORT;

    while (p->running) {
        ASI_ERROR_CODE ret = ASIGetVideoData(ASI_CAM_ID, asi_buf, (long)buf_sz, to_ms);
        if (ret != ASI_SUCCESS) { fprintf(stderr,"[Capture] get err=%d\n",ret); continue; }

        raw_buf_t *raw = (raw_buf_t *)malloc(sizeof(raw_buf_t));
        raw->data = (float *)malloc(4*PLANE_SIZE*sizeof(float));
        raw->frame_id = fid++;
        raw->ts_us = now_us();
        raw->exposure = (expo_seq == 1) ? EXP_LONG : EXP_SHORT;

        float *r=raw->data, *gr=r+PLANE_SIZE, *gb=gr+PLANE_SIZE, *b=gb+PLANE_SIZE;
        bayer_extract((const uint16_t*)asi_buf, asi_img_w, asi_img_h, r, gr, gb, b, cx, cy, bayer_pat, maxval);
        ring_put(&p->raw_ring, &raw);

        /* 交替曝光: S → L → S ... */
        expo_seq = (expo_seq + 1) % 3;
        int new_expo = (expo_seq == 1) ? EXP_LONG : EXP_SHORT;
        if (expo_seq == 1) {
            asi_set_exposure_gain(l_exp, l_gain);
            to_ms = l_exp/1000*2 + 2000;
        } else {
            asi_set_exposure_gain(s_exp, s_gain);
            to_ms = s_exp/1000*2 + 2000;
        }
        /* 仅曝光类型改变时丢弃 1 帧 (S↔L 切换), S→S 不丢 */
        if (new_expo != prev_expo) {
            ASIGetVideoData(ASI_CAM_ID, asi_buf, (long)buf_sz, to_ms);
            prev_expo = new_expo;
        }
    }

    free(asi_buf); asi_cleanup();
    return NULL;
}
