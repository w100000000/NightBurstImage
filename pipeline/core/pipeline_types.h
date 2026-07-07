/*
 * pipeline_types.h — 共享数据结构
 *
 * NBINet 272x480 模型:
 *   输入 Bayer 544x960 → 4-plane RGGB 各 272x480 → NHWC [1,272,480,C]
 *   输出 RGB 544x960 float32 NCHW
 *
 * 线程架构:
 *   Capture → [RAW triple-buf] → NPU → [RGB triple-buf] ──→ Display
 *                                                    └──→ LPR
 */
#ifndef PIPELINE_TYPES_H
#define PIPELINE_TYPES_H

#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

/* ── 分辨率 (模型维度) ── */
#define PLANE_H     272
#define PLANE_W     480
#define PLANE_SIZE  (PLANE_H * PLANE_W)          /* 130560 */
#define BAYER_H     (PLANE_H * 2)                 /* 544 */
#define BAYER_W     (PLANE_W * 2)                 /* 960 */
#define RGB_H       (PLANE_H * 2)                 /* 544 — 模型输出2倍上采样 */
#define RGB_W       (PLANE_W * 2)                 /* 960 */
#define FB_W        1024
#define FB_H        600
#define FEAT_C      4      /* R,Gr,Gb,B 4 plane */
#define EXP_SHORT   0
#define EXP_LONG    1

/* Bayer = BGGR → 提取为 RGGB plane 次序 */
/* ── NPU 输入 ── */
#define SHORT_CAT_C (FEAT_C * 2)  /* 8 */
#define LONG_C      FEAT_C        /* 4 */

/* ── 环形缓冲 ── */
#define RING_SIZE    4

/* ── 时间 ── */
static inline int64_t now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/* ── RAW 4-plane float (Bayer 提取后) ── */
typedef struct {
    float    r[PLANE_H][PLANE_W];   /* 注意: 这些是栈上大数组, 实际用 malloc */
    float    gb[PLANE_H][PLANE_W];   /* 实际 malloc 分配 */
    float    gr[PLANE_H][PLANE_W];
    float    b[PLANE_H][PLANE_W];
} raw_planes_t;

/* 实际用的帧缓冲 */
typedef struct {
    float   *data;          /* R,Gr,Gb,B 连续: 4 * PLANE_H * PLANE_W floats */
    int64_t  ts_us;
    int      exposure;      /* EXP_SHORT or EXP_LONG */
    int      frame_id;
} raw_buf_t;

/* ── NPU 输入 (打包好的 INT8) ── */
typedef struct {
    int8_t  *data;          /* [1,PLANE_H,PLANE_W,C] NHWC, C=4 or 8 */
    int64_t  ts_us;
    int      frame_id;
} npu_input_t;

/* ── RGB 帧 (NPU 输出) ── */
typedef struct {
    float   *data;          /* NCHW [3][RGB_H][RGB_W], 连续 float */
    int64_t  ts_us;
    int      frame_id;
} rgb_buf_t;

/* ── 通用环形缓冲 ── */
typedef struct {
    void          **slots;
    int             size;
    size_t          elem_size;
    int             head, tail;
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
    volatile bool   quit;
} ring_buf_t;

/* ── Ring buffer API ── */
#ifdef __cplusplus
extern "C" {
#endif
int  ring_init(ring_buf_t *rb, int size, size_t elem_size);
void ring_destroy(ring_buf_t *rb);
int  ring_put(ring_buf_t *rb, void *elem);
void* ring_get(ring_buf_t *rb);
int  ring_try_put(ring_buf_t *rb, void *elem);
void* ring_try_get(ring_buf_t *rb);
void ring_stop(ring_buf_t *rb);
#ifdef __cplusplus
}
#endif

/* ── 全局 pipeline 状态 ── */
typedef struct {
    ring_buf_t  raw_ring;      /* raw_buf_t */
    ring_buf_t  rgb_ring;      /* rgb_buf_t, Display 专用 */
    ring_buf_t  lpr_ring;      /* rgb_buf_t, LPR 专用 */

    pthread_t   cap_thread;
    pthread_t   npu_thread;
    pthread_t   dsp_thread;
    pthread_t   lpr_thread;

    bool    no_lpr;
    bool    no_display;
    volatile bool running;

    /* 统计 */
    volatile int64_t frame_cnt;
    volatile int64_t drop_cnt;
    volatile double  npu_ms;
    volatile double  lpr_ms;

    /* NPU 模型信息 */
    int     out_fmt;        /* RKNN_TENSOR_NCHW or NHWC */
    float  *test_frame;     /* 测试: 直传 NPU 输出给 display */
    volatile bool test_ready;

    /* 配置 */
    int     expo_short_us;
    int     expo_long_us;
    char    model_path[256];
    char    lpr_model_dir[256];
    char    storage_dir[256];
} pipeline_t;

#endif /* PIPELINE_TYPES_H */
