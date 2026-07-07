/*
 * npu_worker.c — 从 raw_ring 取3帧 (short,long,short) 拼成 NPU 输入, 推理,
 *               输出 rgb_buf_t 推入 rgb_ring
 */
#include "pipeline_types.h"
#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 打包 4-plane float → float32 NHWC, 步长=total_ch (FP16模型输入需float) */
static void pack_4ch_f(const float *data,   /* [4][PLANE_H][PLANE_W] */
                       float *out,           /* NHWC, 步长 total_ch */
                       int total_ch,         /* 该输入总通道数 (short=8, long=4) */
                       int ch_offset)        /* 写入起始 channel */
{
    const float *r  = data;
    const float *gr = r  + PLANE_SIZE;
    const float *gb = gr + PLANE_SIZE;
    const float *b  = gb + PLANE_SIZE;

    for (int i = 0; i < PLANE_SIZE; i++) {
        int base = i * total_ch + ch_offset;
        out[base + 0] = r[i];
        out[base + 1] = gr[i];
        out[base + 2] = gb[i];
        out[base + 3] = b[i];
    }
}

/* 打包 short_cat: 2帧 short → float [1,PLANE_H,PLANE_W,8]
 * 修复原 bug: 原 pack_4ch 用 FEAT_C=4 步长打包 8 通道, 导致 s1 覆盖 s0 且后半缓冲未初始化 */
static float* make_short_cat(const raw_buf_t *s0, const raw_buf_t *s1)
{
    float *out = (float *)malloc(PLANE_H * PLANE_W * SHORT_CAT_C * sizeof(float));
    pack_4ch_f(s0->data, out, SHORT_CAT_C, 0);  /* short0 → ch[0..3] */
    pack_4ch_f(s1->data, out, SHORT_CAT_C, 4);  /* short1 → ch[4..7] */
    return out;
}

/* 打包 long_raw: float [1,PLANE_H,PLANE_W,4] */
static float* make_long_raw(const raw_buf_t *l)
{
    float *out = (float *)malloc(PLANE_H * PLANE_W * LONG_C * sizeof(float));
    pack_4ch_f(l->data, out, LONG_C, 0);
    return out;
}

/* 原地翻转 RGB (NCHW [3][H][W]): 修正相机上下颠倒/镜像 */
static void flip_rgb(float *d, int H, int W, int fv, int fh)
{
    for (int c = 0; c < 3; c++) {
        float *ch = d + (size_t)c * H * W;
        if (fv) {
            for (int y = 0; y < H/2; y++) {
                float *r0 = ch + y*W, *r1 = ch + (H-1-y)*W;
                for (int x = 0; x < W; x++) { float t = r0[x]; r0[x] = r1[x]; r1[x] = t; }
            }
        }
        if (fh) {
            for (int y = 0; y < H; y++) {
                float *r = ch + y*W;
                for (int x = 0; x < W/2; x++) { float t = r[x]; r[x] = r[W-1-x]; r[W-1-x] = t; }
            }
        }
    }
}

void *npu_thread(void *arg)
{
    pipeline_t *p = (pipeline_t *)arg;

    /* ── 加载模型 ── */
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (void *)p->model_path, 0, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "[NPU] rknn_init %s fail ret=%d\n", p->model_path, ret);
        return NULL;
    }

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("[NPU] model loaded: %s (%d in, %d out)\n",
           p->model_path, io_num.n_input, io_num.n_output);

    /* 查输出属性 */
    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    printf("[NPU] output: size=%d bytes, dims=%d, fmt=%s\n",
           out_attr.size, out_attr.n_dims,
           out_attr.fmt == RKNN_TENSOR_NCHW ? "NCHW" :
           out_attr.fmt == RKNN_TENSOR_NHWC ? "NHWC" : "OTHER");
    p->out_fmt = out_attr.fmt;

    /* 查输入 tensor 量化参数 */
    for (int i = 0; i < io_num.n_input; i++) {
        rknn_tensor_attr in_attr;
        memset(&in_attr, 0, sizeof(in_attr));
        in_attr.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
        printf("[NPU] input[%d]: dims=%d size=%d fmt=%d qnt_type=%d scale=%f zp=%d\n",
               i, in_attr.n_dims, in_attr.size, in_attr.fmt,
               in_attr.qnt_type, in_attr.scale, in_attr.zp);
    }

    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;  inputs[0].type = RKNN_TENSOR_FLOAT32;  inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[1].index = 1;  inputs[1].type = RKNN_TENSOR_FLOAT32;  inputs[1].fmt = RKNN_TENSOR_NHWC;

    while (p->running) {
        /* ── 取 3 帧: short, long, short ── */
        raw_buf_t *s0 = (raw_buf_t *)ring_get(&p->raw_ring);
        if (!s0) break;
        raw_buf_t *l  = (raw_buf_t *)ring_get(&p->raw_ring);
        if (!l) { free(s0->data); free(s0); break; }
        raw_buf_t *s1 = (raw_buf_t *)ring_get(&p->raw_ring);
        if (!s1) { free(s0->data); free(s0); free(l->data); free(l); break; }

        /* ── 打包 (float32) ── */
        float *sc = make_short_cat(s0, s1);
        float *lr = make_long_raw(l);

        inputs[0].buf = sc;
        inputs[0].size = PLANE_H * PLANE_W * SHORT_CAT_C * sizeof(float);
        inputs[1].buf = lr;
        inputs[1].size = PLANE_H * PLANE_W * LONG_C * sizeof(float);

        /* ── 推理 ── */
        int64_t t0 = now_us();
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        ret = rknn_run(ctx, NULL);
        int64_t t1 = now_us();
        if (ret < 0) { fprintf(stderr, "[NPU] rknn_run fail ret=%d\n", ret); }

        /* ── 取输出 ── */
        rknn_output outputs[1];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        outputs[0].index = 0;
        rknn_outputs_get(ctx, 1, outputs, NULL);

        /* ── 拷贝输出 (want_float=1 → float32, out_attr.size 是FP16原生大小需×2) ── */
        rgb_buf_t *rgb = (rgb_buf_t *)malloc(sizeof(rgb_buf_t));
        size_t out_bytes = (size_t)out_attr.n_elems * sizeof(float);
        rgb->data     = (float *)malloc(out_bytes);
        memcpy(rgb->data, outputs[0].buf, out_bytes);
        rgb->ts_us    = l->ts_us;
        rgb->frame_id = p->frame_cnt++;

        /* 修正相机180°倒装 (FLIP_V/FLIP_H默认都1). 在clone前做, 显示+LPR都校正 */
        static int flip_v = -1, flip_h = 1;
        if (flip_v < 0) {
            const char *e = getenv("FLIP_V"); flip_v = e ? atoi(e) : 1;
            e = getenv("FLIP_H"); flip_h = e ? atoi(e) : 1;
            printf("[NPU] flip_v=%d flip_h=%d\n", flip_v, flip_h);
        }
        if (flip_v || flip_h) flip_rgb(rgb->data, RGB_H, RGB_W, flip_v, flip_h);

        rknn_outputs_release(ctx, 1, outputs);

        /* ── clone 一份给 LPR ring ── */
        rgb_buf_t *lpr_copy = (rgb_buf_t *)malloc(sizeof(rgb_buf_t));
        lpr_copy->data     = (float *)malloc(out_bytes);
        memcpy(lpr_copy->data, rgb->data, out_bytes);
        lpr_copy->ts_us    = rgb->ts_us;
        lpr_copy->frame_id = rgb->frame_id;

        /* ── 推 ring: Display 有需求才推, LPR 有需求才推 ── */
        int d_ok = 1, l_ok = 1;
        if (!p->no_display) {
            d_ok = ring_put(&p->rgb_ring, &rgb);
            if (d_ok < 0) { free(rgb->data); free(rgb); }
        } else {
            /* --no-display: 没人消费 rgb_ring, 不推, 直接释放 */
            free(rgb->data); free(rgb);
        }
        if (!p->no_lpr) {
            l_ok = ring_put(&p->lpr_ring, &lpr_copy);
            if (l_ok < 0) { free(lpr_copy->data); free(lpr_copy); }
        } else {
            free(lpr_copy->data); free(lpr_copy);
        }
        if ((!p->no_display && d_ok < 0) || (!p->no_lpr && l_ok < 0)) p->drop_cnt++;

        /* 更新统计 */
        p->npu_ms = (t1 - t0) / 1000.0;

        /* 释放 RAW */
        free(sc); free(lr);
        free(s0->data); free(s0);
        free(l->data);  free(l);
        free(s1->data); free(s1);

        printf("[NPU] frame %lld: %.2f ms\n", (long long)p->frame_cnt, p->npu_ms);
    }

    rknn_destroy(ctx);
    return NULL;
}
