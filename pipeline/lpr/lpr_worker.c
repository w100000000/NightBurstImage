/*
 * lpr_worker.c — CPU 车牌识别 (ncnn) 完整实现
 *
 * Pipeline: rgb_ring → YOLO detect → crop ROI → CTC recognize → classify → log
 *
 * 基于 HyperLPR3 (https://github.com/szad670401/HyperLPR):
 *   - YOLO anchors/decode 来自 Prj-Python/hyperlpr3/common/tools_process.py
 *   - CTC alphabet 来自 Prj-Python/hyperlpr3/common/tokenize.py
 *   - 预处理参数 来自 cpp/src/nn_implementation_module/
 *
 * 依赖: libncnn.a
 */
#include "pipeline_types.h"
#include "net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* ══════════════════════════════════════════════════════════════════════
 *  HyperLPR3 常量
 * ══════════════════════════════════════════════════════════════════════ */

/* ── YOLO detection ── */
#define YOLO_IN_W       320
#define YOLO_IN_H       320
#define YOLO_CONF       0.35f
#define YOLO_NMS        0.5f
#define MAX_DET          20

/* anchors for 320 input (9 pairs, grouped by mask [[0,1,2],[3,4,5],[6,7,8]]) */
static const float YOLO_ANCHORS[9][2] = {
    {9.38281f,  3.08398f}, {15.53125f, 4.93750f}, {19.98438f, 7.78906f},
    {31.10938f, 10.35156f}, {45.21875f, 14.14844f}, {32.34375f, 21.04688f},
    {65.62500f, 19.57812f}, {76.12500f, 46.12500f}, {253.25000f, 137.50000f}
};
static const int YOLO_MASKS[3][3] = {{0,1,2}, {3,4,5}, {6,7,8}};
static const int YOLO_GRIDS[3][2]  = {{40,40}, {20,20}, {10,10}};
static const int YOLO_STRIDES[3]   = {8, 16, 32};

/* ── CTC recognition ── */
#define CTC_IN_W        160
#define CTC_IN_H         48
#define CTC_OUT_T        20    /* time steps (rpv3模型实际输出[1,20,78], 原写40会读越界出乱码尾巴) */
#define CTC_NCLASS        78   /* blank + 77 chars */
#define CTC_BLANK          0

static const char *CTC_ALPHABET[78] = {
    "[blank]", "'", "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","J","K","L","M","N",
    "O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "\xe4\xba\x91",  /*  云 */  "\xe4\xba\xac",  /*  京 */
    "\xe5\x86\x80",  /*  冀 */  "\xe5\x90\x89",  /*  吉 */
    "\xe5\xad\xa6",  /*  学 */  "\xe5\xae\x81",  /*  宁 */
    "\xe5\xb7\x9d",  /*  川 */  "\xe6\x8c\x82",  /*  挂 */
    "\xe6\x96\xb0",  /*  新 */  "\xe6\x99\x8b",  /*  晋 */
    "\xe6\xa1\x82",  /*  桂 */  "\xe6\xb0\x91",  /*  民 */
    "\xe6\xb2\xaa",  /*  沪 */  "\xe6\xb4\xa5",  /*  津 */
    "\xe6\xb5\x99",  /*  浙 */  "\xe6\xb8\x9d",  /*  渝 */
    "\xe6\xb8\xaf",  /*  港 */  "\xe6\xb9\x98",  /*  湘 */
    "\xe7\x90\xbc",  /*  琼 */  "\xe7\x94\x98",  /*  甘 */
    "\xe7\x9a\x96",  /*  皖 */  "\xe7\xb2\xa4",  /*  粤 */
    "\xe8\x88\xaa",  /*  航 */  "\xe8\x8b\x8f",  /*  苏 */
    "\xe8\x92\x99",  /*  蒙 */  "\xe8\x97\x8f",  /*  藏 */
    "\xe8\xad\xa6",  /*  警 */  "\xe8\xb1\xab",  /*  豫 */
    "\xe8\xb4\xb5",  /*  贵 */  "\xe8\xb5\xa3",  /*  赣 */
    "\xe8\xbe\xbd",  /*  辽 */  "\xe9\x84\x82",  /*  鄂 */
    "\xe9\x97\xbd",  /*  闽 */  "\xe9\x99\x95",  /*  陕 */
    "\xe9\x9d\x92",  /*  青 */  "\xe9\xb2\x81",  /*  鲁 */
    "\xe9\xbb\x91",  /*  黑 */  "\xe9\xa2\x86",  /*  领 */
    "\xe4\xbd\xbf",  /*  使 */  "\xe6\xbe\xb3",  /*  澳 */
};

/* ── Classification ── */
#define CLS_IN_W         96
#define CLS_IN_H         96
#define CLS_NCLASS        3

static const char *CLS_LABELS[3] = {"Blue", "Green", "Yellow"};

/* ══════════════════════════════════════════════════════════════════════
 *  Math helpers
 * ══════════════════════════════════════════════════════════════════════ */

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float clip_f(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Bilinear resize (CHW float)
 * ══════════════════════════════════════════════════════════════════════ */

static void resize_chw(const float *src, int sh, int sw,
                       float *dst, int dh, int dw)
{
    float sy = (float)sh / dh, sx = (float)sw / dw;
    for (int c = 0; c < 3; c++) {
        const float *sc = src + c * sh * sw;
        float       *dc = dst + c * dh * dw;
        for (int y = 0; y < dh; y++) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = (int)fy; if (y0 < 0) y0 = 0; if (y0 >= sh-1) y0 = sh-2;
            int y1 = y0 + 1;
            float dy = fy - y0;
            for (int x = 0; x < dw; x++) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = (int)fx; if (x0 < 0) x0 = 0; if (x0 >= sw-1) x0 = sw-2;
                int x1 = x0 + 1;
                float dx = fx - x0;
                dc[y*dw+x] = (1-dy)*((1-dx)*sc[y0*sw+x0] + dx*sc[y0*sw+x1])
                           +    dy *((1-dx)*sc[y1*sw+x0] + dx*sc[y1*sw+x1]);
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  抓拍存图: CHW BGR float[0,1] → BMP (无依赖, 浏览器可显示)
 *  gamma 提亮(去噪原图偏暗), 底部朝上写行
 * ══════════════════════════════════════════════════════════════════════ */
static void bmp_write_chw_bgr(const char *path, const float *chw, int w, int h)
{
    int rowsz = (w * 3 + 3) & ~3;      /* 行4字节对齐 */
    int imgsz = rowsz * h;
    int filesz = 54 + imgsz;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    unsigned char hdr[54];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    hdr[2]=filesz&0xff; hdr[3]=(filesz>>8)&0xff; hdr[4]=(filesz>>16)&0xff; hdr[5]=(filesz>>24)&0xff;
    hdr[10]=54; hdr[14]=40;
    hdr[18]=w&0xff; hdr[19]=(w>>8)&0xff; hdr[20]=(w>>16)&0xff; hdr[21]=(w>>24)&0xff;
    hdr[22]=h&0xff; hdr[23]=(h>>8)&0xff; hdr[24]=(h>>16)&0xff; hdr[25]=(h>>24)&0xff;
    hdr[26]=1; hdr[28]=24;
    fwrite(hdr, 1, 54, f);

    const float *B = chw, *G = chw + w*h, *R = chw + 2*w*h;
    unsigned char *row = (unsigned char *)malloc(rowsz);
    for (int y = h - 1; y >= 0; y--) {          /* BMP 自底向上 */
        memset(row, 0, rowsz);
        for (int x = 0; x < w; x++) {
            int i = y*w + x;
            /* gamma 0.6 提亮 */
            int b = (int)(powf(B[i] > 0 ? B[i] : 0, 0.6f) * 255 + 0.5f);
            int g = (int)(powf(G[i] > 0 ? G[i] : 0, 0.6f) * 255 + 0.5f);
            int r = (int)(powf(R[i] > 0 ? R[i] : 0, 0.6f) * 255 + 0.5f);
            if (b > 255) b = 255; if (g > 255) g = 255; if (r > 255) r = 255;
            row[x*3+0] = b; row[x*3+1] = g; row[x*3+2] = r;
        }
        fwrite(row, 1, rowsz, f);
    }
    free(row);
    fclose(f);
}

/* ══════════════════════════════════════════════════════════════════════
 *  YOLO 后处理: 3-grid raw features → decoded boxes → NMS
 *  完全对照 HyperLPR3 Python tools_process.py / detect.py
 * ══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float x1, y1, x2, y2;   /* image coordinates */
    float conf;
    int   klass;             /* layer: 0=single, 1=double */
} bbox_t;

/* YOLO raw output: 3 tensors, each [3, 6, H, W] reshaped to [H, W, 3, 6]
   where 6 = (x, y, w, h, conf, class_prob) */
static int yolo_decode_raw(const float *feat, int gh, int gw,
                            int stride, int mask_idx,
                            bbox_t *boxes, int max_box)
{
    int count = 0;
    int na = 3; /* 3 anchors per grid */

    for (int gy = 0; gy < gh && count < max_box; gy++) {
        for (int gx = 0; gx < gw && count < max_box; gx++) {
            for (int a = 0; a < na && count < max_box; a++) {
                int ai = YOLO_MASKS[mask_idx][a];
                float aw = YOLO_ANCHORS[ai][0];
                float ah = YOLO_ANCHORS[ai][1];

                /* feat layout: [H, W, 3_anchors, 6_values] */
                int off = ((gy * gw + gx) * na + a) * 6;

                float bx = sigmoid_f(feat[off + 0]) * 2.0f - 0.5f + (float)gx;
                float by = sigmoid_f(feat[off + 1]) * 2.0f - 0.5f + (float)gy;
                float bw = powf(sigmoid_f(feat[off + 2]) * 2.0f, 2) * aw;
                float bh = powf(sigmoid_f(feat[off + 3]) * 2.0f, 2) * ah;

                float obj_conf  = sigmoid_f(feat[off + 4]);
                float cls_prob  = sigmoid_f(feat[off + 5]);
                float score     = obj_conf * cls_prob;

                if (score < YOLO_CONF) continue;

                /* convert to image coordinates (cxcywh in 320x320 space) */
                float cx = bx * stride;
                float cy = by * stride;
                float w  = bw * stride;
                float h  = bh * stride;

                boxes[count].x1   = cx - w/2;
                boxes[count].y1   = cy - h/2;
                boxes[count].x2   = cx + w/2;
                boxes[count].y2   = cy + h/2;
                boxes[count].conf = score;
                boxes[count].klass = cls_prob > 0.5f ? 1 : 0;
                count++;
            }
        }
    }
    return count;
}

/* ── NMS (IoU-based) ── */
static int nms_boxes(bbox_t *boxes, int n, float thresh)
{
    if (n <= 0) return 0;

    /* sort by confidence descending */
    int *order = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (boxes[order[j]].conf > boxes[order[i]].conf) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }

    int keep_count = 0;
    int *keep = (int *)calloc(n, sizeof(int));

    for (int i = 0; i < n; i++) {
        int idx = order[i];
        if (keep[idx]) continue; /* already suppressed */

        keep[idx] = 1;
        keep_count++;

        float bx1 = boxes[idx].x1, by1 = boxes[idx].y1;
        float bx2 = boxes[idx].x2, by2 = boxes[idx].y2;
        float barea = (bx2 - bx1) * (by2 - by1);

        for (int j = i + 1; j < n; j++) {
            int jdx = order[j];
            if (keep[jdx]) continue;

            float ix1 = fmaxf(bx1, boxes[jdx].x1);
            float iy1 = fmaxf(by1, boxes[jdx].y1);
            float ix2 = fminf(bx2, boxes[jdx].x2);
            float iy2 = fminf(by2, boxes[jdx].y2);
            float iw  = fmaxf(0, ix2 - ix1 + 0.00001f);
            float ih  = fmaxf(0, iy2 - iy1 + 0.00001f);
            float inter = iw * ih;

            float jarea = (boxes[jdx].x2 - boxes[jdx].x1) * (boxes[jdx].y2 - boxes[jdx].y1);
            float iou = inter / (barea + jarea - inter);

            if (iou > thresh) keep[jdx] = 1; /* suppress */
        }
    }

    /* compact: move kept boxes to front */
    int out = 0;
    for (int i = 0; i < n; i++) {
        if (keep[i] && out < i) boxes[out] = boxes[i];
        if (keep[i]) out++;
    }

    free(order); free(keep);
    return out;
}

/* ══════════════════════════════════════════════════════════════════════
 *  CTC 解码: argmax → skip blank → collapse duplicates
 *  对照 HyperLPR3 recognition.py decode()
 * ══════════════════════════════════════════════════════════════════════ */

static int ctc_decode(const float *output, int T, int nclass,
                      int *tokens, int max_tokens)
{
    int nt = 0;
    int prev = -1;

    for (int t = 0; t < T && nt < max_tokens; t++) {
        int best = argmax(output + t * nclass, nclass);

        if (best == CTC_BLANK) {   /* skip blank */
            prev = -1;
            continue;
        }
        if (best == prev) continue; /* collapse duplicates */
        prev = best;
        tokens[nt++] = best;
    }
    return nt;
}

/* ══════════════════════════════════════════════════════════════════════
 *  LPR 主线程
 * ══════════════════════════════════════════════════════════════════════ */

extern "C" void *lpr_thread(void *arg)
{
    pipeline_t *p = (pipeline_t *)arg;

    /* ── 加载 3 个 ncnn 模型 ── */
    char path[512];
    ncnn::Net yolo_net, ctc_net, cls_net;

    /* YOLO */
    snprintf(path, sizeof(path), "%s/y5fu_320x_sim.ncnn.param", p->lpr_model_dir);
    if (yolo_net.load_param(path) != 0) {
        fprintf(stderr, "[LPR] FAIL: load YOLO param %s\n", path);
        return NULL;
    }
    snprintf(path, sizeof(path), "%s/y5fu_320x_sim.ncnn.bin", p->lpr_model_dir);
    if (yolo_net.load_model(path) != 0) {
        fprintf(stderr, "[LPR] FAIL: load YOLO bin %s\n", path);
        return NULL;
    }

    /* CTC */
    snprintf(path, sizeof(path), "%s/rpv3_mdict_160_r3.ncnn.param", p->lpr_model_dir);
    if (ctc_net.load_param(path) != 0) {
        fprintf(stderr, "[LPR] FAIL: load CTC param %s\n", path);
        return NULL;
    }
    snprintf(path, sizeof(path), "%s/rpv3_mdict_160_r3.ncnn.bin", p->lpr_model_dir);
    ctc_net.load_model(path);

    /* CLS */
    snprintf(path, sizeof(path), "%s/litemodel_cls_96x_r1.ncnn.param", p->lpr_model_dir);
    if (cls_net.load_param(path) != 0) {
        fprintf(stderr, "[LPR] FAIL: load CLS param %s\n", path);
        return NULL;
    }
    snprintf(path, sizeof(path), "%s/litemodel_cls_96x_r1.ncnn.bin", p->lpr_model_dir);
    cls_net.load_model(path);

    /* ── 输出模型信息 ── */
    {
        auto &y_in  = yolo_net.input_names();
        auto &y_out = yolo_net.output_names();
        printf("[LPR] YOLO: %zu in, %zu out\n", y_in.size(), y_out.size());
        for (size_t i = 0; i < y_out.size(); i++)
            printf("       out[%zu]: %s\n", i, y_out[i]);
    }

    printf("[LPR] models loaded from %s\n", p->lpr_model_dir);

    /* ── 分配临时缓冲 ── */
    float *yolo_in = (float *)malloc(3 * YOLO_IN_H * YOLO_IN_W * sizeof(float));

    /* 存储目录 */
    mkdir(p->storage_dir, 0755);

    /* ── 主循环 ── */
    while (p->running) {
        rgb_buf_t *rgb = (rgb_buf_t *)ring_get(&p->lpr_ring);
        if (!rgb) break;

        int64_t t0 = now_us();
        int n_plates = 0;

        /* ─── Step 1: YOLO 检测 ─── */
        /* resize 去噪 RGB [3,RGB_H,RGB_W] → YOLO 输入 [3,320,320] */
        resize_chw(rgb->data, RGB_H, RGB_W, yolo_in, YOLO_IN_H, YOLO_IN_W);

        /* pack into ncnn::Mat (WHC format) */
        bbox_t boxes[MAX_DET];
        int nbox = 0;

        {
            ncnn::Mat in(YOLO_IN_W, YOLO_IN_H, 3);
            /* yolo_in 已是 [0,1] (NPU输出), YOLO 期望 [0,1], 不做归一 (原代码误 /255 ≈全黑) */
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < YOLO_IN_H; y++)
                    for (int x = 0; x < YOLO_IN_W; x++)
                        in.channel(c)[y * YOLO_IN_W + x] =
                            yolo_in[c * YOLO_IN_H * YOLO_IN_W + y * YOLO_IN_W + x];
            }

            ncnn::Extractor ex = yolo_net.create_extractor();
            ex.input(yolo_net.input_names()[0], in);

            /* probe output format */
            auto &out_names = yolo_net.output_names();
            int n_out = (int)out_names.size();

            if (n_out == 3) {
                /* ── Format A: 3 raw feature maps, each [18, H, W] ncnn CHW ── */
                for (int g = 0; g < 3; g++) {
                    ncnn::Mat out;
                    ex.extract(out_names[g], out);
                    int C = out.c, H = out.h, W = out.w;
                    int na = 3, nv = C / na; /* nv = 6 (x,y,w,h,conf,cls) */

                    /* Pack into [H, W, na, nv] float array */
                    float *dec = (float *)malloc(H * W * na * nv * sizeof(float));
                    for (int y = 0; y < H; y++) {
                        for (int x = 0; x < W; x++) {
                            for (int a = 0; a < na; a++) {
                                for (int v = 0; v < nv; v++) {
                                    /* ncnn CHW: channel(a*nv+v)[y][x] */
                                    dec[((y*W + x)*na + a)*nv + v] =
                                        ((float*)out.channel(a*nv + v))[y*W + x];
                                }
                            }
                        }
                    }

                    int n = yolo_decode_raw(dec, H, W, YOLO_STRIDES[g], g,
                                            boxes + nbox, MAX_DET - nbox);
                    nbox += n;
                    free(dec);
                }
            } else {
                /* ── Format B: single flat pre-decoded tensor [6300, 15] ──
                 * PNNX 已做: sigmoid(xy/conf/cls), anchor*grid decode
                 * 15 值/cell: [0:2]=cx,cy [2:4]=w,h [4]=obj_conf [5:13]=landmarks [13:15]=cls
                 * 注意: conf 是原始 float (非 logit), 直接比较阈值, 不需要 sigmoid */
                ncnn::Mat out;
                ex.extract(out_names[0], out);
                int total = out.w * out.h * out.c;
                float *flat = (float *)out.data;

                printf("[LPR] YOLO out: w=%d h=%d c=%d total=%d\n", out.w, out.h, out.c, total);

                int item_num;
                if (total == 6300 * 15)      item_num = 15;
                else if (total == 6300 * 6)  item_num = 6;
                else if (total == 25200 * 15) item_num = 15;
                else if (total == 25200 * 6)  item_num = 6;
                else {
                    item_num = (total % 6300 == 0) ? 15 : 6;
                    printf("[LPR] WARNING: unknown YOLO shape total=%d, item_num=%d\n", total, item_num);
                }

                int grid_num = total / item_num;

                for (int i = 0; i < grid_num && nbox < MAX_DET; i++) {
                    /* PNNX model output: conf at [4] is raw float, NOT a logit */
                    float conf = flat[i * item_num + 4];

                    if (conf < YOLO_CONF) continue;

                    /* xy/wh 已被 PNNX decode 到 320x320 图像坐标 */
                    float cx = flat[i * item_num + 0];
                    float cy = flat[i * item_num + 1];
                    float w  = flat[i * item_num + 2];
                    float h  = flat[i * item_num + 3];

                    boxes[nbox].x1   = cx - w/2;
                    boxes[nbox].y1   = cy - h/2;
                    boxes[nbox].x2   = cx + w/2;
                    boxes[nbox].y2   = cy + h/2;
                    boxes[nbox].conf = conf;
                    boxes[nbox].klass = 0;
                    nbox++;
                }
            }
        }

        /* NMS across all grid outputs */
        nbox = nms_boxes(boxes, nbox, YOLO_NMS);

        int64_t t_yolo = now_us();

        /* 单车牌场景: 只保留置信度最高的框, 避免同一车牌被多个框重复识别/打印 */
        if (nbox > 1) {
            int best = 0;
            for (int i = 1; i < nbox; i++)
                if (boxes[i].conf > boxes[best].conf) best = i;
            boxes[0] = boxes[best];
            nbox = 1;
        }

        /* ─── Step 2: 对每个检测框做 CTC 识别 ─── */
        for (int i = 0; i < nbox; i++) {
            bbox_t *b = &boxes[i];

            /* map YOLO coords (320x320) back to original RGB image (RGB_W × RGB_H) */
            float scale_x = (float)RGB_W / YOLO_IN_W;
            float scale_y = (float)RGB_H / YOLO_IN_H;
            int x1 = (int)(b->x1 * scale_x);
            int y1 = (int)(b->y1 * scale_y);
            int x2 = (int)(b->x2 * scale_x);
            int y2 = (int)(b->y2 * scale_y);

            /* clip */
            if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
            if (x2 >= RGB_W) x2 = RGB_W - 1;
            if (y2 >= RGB_H) y2 = RGB_H - 1;
            if (x2 <= x1 || y2 <= y1) continue;

            /* crop ROI from rgb->data [C, H, W] */
            int crop_w = x2 - x1, crop_h = y2 - y1;
            float *roi = (float *)malloc(3 * crop_h * crop_w * sizeof(float));
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < crop_h; y++)
                    memcpy(roi + c*crop_h*crop_w + y*crop_w,
                           rgb->data + c*RGB_H*RGB_W + (y1+y)*RGB_W + x1,
                           crop_w * sizeof(float));

            /* ── CTC recognition ── */
            {
                /* resize ROI to 48x160, normalize (x-127.5)/127.5 */
                /* rgb->data is [0,1], so first scale to [0,255] */
                float *ctc_in = (float *)calloc(3 * CTC_IN_H * CTC_IN_W, sizeof(float));
                float *roi_255 = (float *)malloc(3 * crop_h * crop_w * sizeof(float));
                for (int j = 0; j < 3 * crop_h * crop_w; j++)
                    roi_255[j] = roi[j] * 255.0f;

                resize_chw(roi_255, crop_h, crop_w, ctc_in, CTC_IN_H, CTC_IN_W);

                /* normalize: (x - 127.5) / 127.5 */
                for (int j = 0; j < 3 * CTC_IN_H * CTC_IN_W; j++)
                    ctc_in[j] = (ctc_in[j] - 127.5f) / 127.5f;

                ncnn::Mat ctc_mat(CTC_IN_W, CTC_IN_H, 3);
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < CTC_IN_H; y++)
                        for (int x = 0; x < CTC_IN_W; x++)
                            ctc_mat.channel(c)[y * CTC_IN_W + x] =
                                ctc_in[c * CTC_IN_H * CTC_IN_W + y * CTC_IN_W + x];

                ncnn::Extractor ex = ctc_net.create_extractor();
                ex.input(ctc_net.input_names()[0], ctc_mat);
                ncnn::Mat ctc_out;
                ex.extract(ctc_net.output_names()[0], ctc_out);

                /* CTC decode: output [CTC_OUT_T, CTC_NCLASS] */
                int tokens[32];
                int nt = ctc_decode((float *)ctc_out.data, CTC_OUT_T, CTC_NCLASS, tokens, 32);

                /* ── Classification ── */
                int plate_type = -1;
                {
                    float *cls_in = (float *)malloc(3 * CLS_IN_H * CLS_IN_W * sizeof(float));
                    resize_chw(roi_255, crop_h, crop_w, cls_in, CLS_IN_H, CLS_IN_W);
                    /* normalize: /255.0 */
                    for (int j = 0; j < 3 * CLS_IN_H * CLS_IN_W; j++)
                        cls_in[j] /= 255.0f;

                    ncnn::Mat cls_mat(CLS_IN_W, CLS_IN_H, 3);
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < CLS_IN_H; y++)
                            for (int x = 0; x < CLS_IN_W; x++)
                                cls_mat.channel(c)[y * CLS_IN_W + x] =
                                    cls_in[c * CLS_IN_H * CLS_IN_W + y * CLS_IN_W + x];

                    ncnn::Extractor ex = cls_net.create_extractor();
                    ex.input(cls_net.input_names()[0], cls_mat);
                    ncnn::Mat cls_out;
                    ex.extract(cls_net.output_names()[0], cls_out);
                    plate_type = argmax((float *)cls_out.data, CLS_NCLASS);

                    free(cls_in);
                }

                /* ── Write result ── */
                /* build plate string from CTC tokens */
                char plate_str[64] = {0};
                int pos = 0;
                for (int t = 0; t < nt && pos < 60; t++) {
                    const char *ch = CTC_ALPHABET[tokens[t]];
                    int len = (int)strlen(ch);
                    if (pos + len < 60) {
                        memcpy(plate_str + pos, ch, len);
                        pos += len;
                    }
                }

                int64_t t_end = now_us();
                double lpr_ms = (t_end - t0) / 1000.0;

                /* 抓拍存图: 去噪全帧缩到 480×272 (省流量, gamma提亮), 限流见下 */
                #define CAP_W 480
                #define CAP_H 272
                char img_name[128] = "";
                static char last_plate[64] = "";
                static long long last_saved = -1000;
                if (b->conf >= 0.5f &&
                    (strcmp(plate_str, last_plate) != 0 ||
                     (long long)rgb->frame_id - last_saved >= 16)) {
                    snprintf(img_name, sizeof(img_name), "cap_%lld.bmp", (long long)rgb->frame_id);
                    char imgpath[700];
                    snprintf(imgpath, sizeof(imgpath), "%s/%s", p->storage_dir, img_name);
                    float *cap = (float *)malloc(3 * CAP_H * CAP_W * sizeof(float));
                    resize_chw(rgb->data, RGB_H, RGB_W, cap, CAP_H, CAP_W);
                    bmp_write_chw_bgr(imgpath, cap, CAP_W, CAP_H);
                    free(cap);
                    strncpy(last_plate, plate_str, 63);
                    last_saved = rgb->frame_id;
                }

                snprintf(path, sizeof(path), "%s/plates.txt", p->storage_dir);
                FILE *fp = fopen(path, "a");
                if (fp) {
                    fprintf(fp, "frame=%lld ts=%lld plate=%s type=%s(%d) "
                            "conf=%.2f img=%s yolo_ms=%.1f lpr_ms=%.1f\n",
                            (long long)rgb->frame_id, (long long)rgb->ts_us,
                            plate_str, plate_type >= 0 ? CLS_LABELS[plate_type] : "?",
                            plate_type, b->conf, img_name,
                            (t_yolo - t0) / 1000.0, lpr_ms);
                    fclose(fp);
                }

                printf("[LPR] frame=%lld plate=%s type=%d conf=%.2f lpr=%.1fms\n",
                       (long long)rgb->frame_id, plate_str, plate_type, b->conf, lpr_ms);

                n_plates++;
                free(ctc_in);
                free(roi_255);
            }
            free(roi);
        }

        p->lpr_ms = (n_plates > 0) ? (now_us() - t0) / 1000.0 : (t_yolo - t0) / 1000.0;

        free(rgb->data);
        free(rgb);
    }

    free(yolo_in);
    printf("[LPR] stopped\n");
    return NULL;
}
