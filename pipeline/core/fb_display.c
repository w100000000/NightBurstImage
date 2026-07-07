/*
 * fb_display.c — NPU RGB输出 → resize → MIPI 屏 (1024x600) 经 DRM/KMS 直显
 *
 * 旧方案(写 /dev/fb0)在本板不可靠: Weston 退出后无 fbcon/无 modeset,
 * VOP 不扫描 fb0。改为 DRM 直控:
 *   1) stop Weston 释放 DRM master
 *   2) 打开 /dev/dri/card0, 找 connected connector(DSI) + mode + CRTC
 *   3) 建 dumb buffer + mmap + drmModeSetCrtc 挂上 (进程持 master, CRTC 常活)
 *   4) 循环把去噪 RGB 直接写进 mapped scanout buffer (pitch 由 DRM 给出)
 */
#include "pipeline_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

/* ── format-aware 取像素 ── */
#define OUT_NCHW 0
#define OUT_NHWC 1

/* 画面增强 (解决"发灰"): 对比度/饱和度, 环境变量 CONTRAST/SAT 可调 (默认轻度增强) */
static float g_contrast = 1.15f;   /* >1 增强对比度, 1.0=off */
static float g_sat      = 1.25f;   /* >1 增强饱和度, 1.0=off */

/* 逐像素增强: 先对比度(绕0.5), 再饱和度(绕亮度) */
static inline void enhance(float *r, float *g, float *b)
{
    if (g_contrast != 1.0f) {
        *r = (*r - 0.5f) * g_contrast + 0.5f;
        *g = (*g - 0.5f) * g_contrast + 0.5f;
        *b = (*b - 0.5f) * g_contrast + 0.5f;
    }
    if (g_sat != 1.0f) {
        float luma = 0.299f * *r + 0.587f * *g + 0.114f * *b;
        *r = luma + (*r - luma) * g_sat;
        *g = luma + (*g - luma) * g_sat;
        *b = luma + (*b - luma) * g_sat;
    }
}

static inline void get_pixel(const float *src, int x, int y,
                             float *r, float *g, float *b, int fmt)
{
    float rr, gg, bb;
    if (fmt == OUT_NCHW) {
        int off = y * RGB_W + x;
        /* 模型输出通道顺序是 BGR! 训练 GT 用 cv2.COLOR_BayerRG2RGB_EA 对 RGGB
         * 转换, 因 OpenCV Bayer 命名偏移实际产出 BGR (ch0=B ch1=G ch2=R),
         * 模型学的就是这个顺序。故 R=ch2, G=ch1, B=ch0。(经 dump 帧数值验证) */
        bb = src[off];                          /* ch0 = B */
        gg = src[RGB_H * RGB_W + off];          /* ch1 = G */
        rr = src[2 * RGB_H * RGB_W + off];      /* ch2 = R */
    } else {
        int off = (y * RGB_W + x) * 3;
        rr = src[off + 2];
        gg = src[off + 1];
        bb = src[off + 0];
    }
    *r = rr; *g = gg; *b = bb;
}

/* ── bilinear resize: NPU output(RGB_W×RGB_H) → 目标(out_w×out_h), 直写 dst ──
 * dst_stride: dst 每行像素数 (= DRM pitch / 4), 保证 stride 正确不花屏 */
static void convert_to_fb(const float *src, uint32_t *dst,
                          int out_w, int out_h, int dst_stride, int fmt)
{
    for (int dy = 0; dy < out_h; dy++) {
        for (int dx = 0; dx < out_w; dx++) {
            float sx = (dx + 0.5f) * RGB_W / (float)out_w;
            float sy = (dy + 0.5f) * RGB_H / (float)out_h;

            if (sx < 0) sx = 0; if (sx >= RGB_W) sx = RGB_W - 1;
            if (sy < 0) sy = 0; if (sy >= RGB_H) sy = RGB_H - 1;

            int x0 = (int)sx, y0 = (int)sy;
            int x1 = x0 + 1; if (x1 >= RGB_W) x1 = x0;
            int y1 = y0 + 1; if (y1 >= RGB_H) y1 = y0;
            float wx = sx - x0, wy = sy - y0;

            float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
            get_pixel(src, x0, y0, &r00, &g00, &b00, fmt);
            get_pixel(src, x1, y0, &r10, &g10, &b10, fmt);
            get_pixel(src, x0, y1, &r01, &g01, &b01, fmt);
            get_pixel(src, x1, y1, &r11, &g11, &b11, fmt);

            float r = (1-wy)*((1-wx)*r00 + wx*r10) + wy*((1-wx)*r01 + wx*r11);
            float g = (1-wy)*((1-wx)*g00 + wx*g10) + wy*((1-wx)*g01 + wx*g11);
            float b = (1-wy)*((1-wx)*b00 + wx*b10) + wy*((1-wx)*b01 + wx*b11);

            enhance(&r, &g, &b);   /* 对比度/饱和度增强, 去灰 */

            if (r > 1.0f) r = 1.0f; if (r < 0) r = 0;
            if (g > 1.0f) g = 1.0f; if (g < 0) g = 0;
            if (b > 1.0f) b = 1.0f; if (b < 0) b = 0;

            int ir = (int)(r * 255.0f + 0.5f);
            int ig = (int)(g * 255.0f + 0.5f);
            int ib = (int)(b * 255.0f + 0.5f);

            /* XRGB8888 (DRM_FORMAT_XRGB8888): 内存字节序 B,G,R,X */
            dst[dy * dst_stride + dx] = (uint32_t)ib | ((uint32_t)ig << 8) |
                                        ((uint32_t)ir << 16);
        }
    }
}

void *display_thread(void *arg)
{
    pipeline_t *p = (pipeline_t *)arg;

    /* 画面增强参数 (环境变量, 方便现场调): CONTRAST=1.15 SAT=1.25 */
    const char *env = getenv("CONTRAST");  if (env) g_contrast = atof(env);
    env = getenv("SAT");                   if (env) g_sat      = atof(env);
    printf("[Display] contrast=%.2f sat=%.2f\n", g_contrast, g_sat);

    /* 1. 停 Weston, 释放 DRM master */
    int ret = system("/etc/init.d/S49weston stop 2>/dev/null");
    (void)ret;
    usleep(500000);

    /* 2. 打开 DRM 设备 */
    int drmfd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drmfd < 0) { perror("[Display] open /dev/dri/card0"); return NULL; }
    drmSetMaster(drmfd);  /* Weston 停后应可获取 master */

    drmModeRes *res = drmModeGetResources(drmfd);
    if (!res) { fprintf(stderr, "[Display] drmModeGetResources fail\n"); close(drmfd); return NULL; }

    /* 找一个 connected 的 connector: 优先 DSI, 否则第一个 connected */
    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector *c = drmModeGetConnector(drmfd, res->connectors[i]);
        if (!c) continue;
        if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            if (c->connector_type == DRM_MODE_CONNECTOR_DSI) { /* 首选 DSI */
                if (conn) drmModeFreeConnector(conn);
                conn = c; break;
            }
            if (!conn) { conn = c; continue; }  /* 暂存第一个 connected */
        }
        drmModeFreeConnector(c);
    }
    if (!conn) { fprintf(stderr, "[Display] no connected connector\n");
                 drmModeFreeResources(res); close(drmfd); return NULL; }

    drmModeModeInfo mode = conn->modes[0];  /* 首选 mode, 通常即 1024x600 */
    printf("[Display] connector=%u type=%d mode=%ux%u@%u\n",
           conn->connector_id, conn->connector_type,
           mode.hdisplay, mode.vdisplay, mode.vrefresh);

    /* 找 CRTC: 用当前 encoder 的 crtc, 否则从 possible_crtcs 选 */
    uint32_t crtc_id = 0;
    if (conn->encoder_id) {
        drmModeEncoder *enc = drmModeGetEncoder(drmfd, conn->encoder_id);
        if (enc) { crtc_id = enc->crtc_id; drmModeFreeEncoder(enc); }
    }
    if (!crtc_id) {
        for (int i = 0; i < conn->count_encoders && !crtc_id; i++) {
            drmModeEncoder *enc = drmModeGetEncoder(drmfd, conn->encoders[i]);
            if (!enc) continue;
            for (int j = 0; j < res->count_crtcs; j++) {
                if (enc->possible_crtcs & (1 << j)) { crtc_id = res->crtcs[j]; break; }
            }
            drmModeFreeEncoder(enc);
        }
    }
    if (!crtc_id) { fprintf(stderr, "[Display] no CRTC found\n");
                    drmModeFreeConnector(conn); drmModeFreeResources(res);
                    close(drmfd); return NULL; }
    printf("[Display] using CRTC %u\n", crtc_id);

    /* 3. 建 dumb buffer (mode 尺寸, XRGB8888 32bpp) */
    struct drm_mode_create_dumb creq;
    memset(&creq, 0, sizeof(creq));
    creq.width  = mode.hdisplay;
    creq.height = mode.vdisplay;
    creq.bpp    = 32;
    if (drmIoctl(drmfd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
        perror("[Display] CREATE_DUMB");
        drmModeFreeConnector(conn); drmModeFreeResources(res); close(drmfd); return NULL;
    }
    int dst_stride = creq.pitch / 4;  /* 每行像素数 = DRM 给的 pitch */
    printf("[Display] dumb buffer: %ux%u pitch=%u (%d px) size=%llu\n",
           creq.width, creq.height, creq.pitch, dst_stride,
           (unsigned long long)creq.size);

    uint32_t fb_id = 0;
    if (drmModeAddFB(drmfd, creq.width, creq.height, 24, 32,
                     creq.pitch, creq.handle, &fb_id) < 0) {
        perror("[Display] drmModeAddFB");
        goto cleanup_dumb;
    }

    struct drm_mode_map_dumb mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.handle = creq.handle;
    if (drmIoctl(drmfd, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) {
        perror("[Display] MAP_DUMB"); goto cleanup_fb;
    }
    uint32_t *vaddr = (uint32_t *)mmap(0, creq.size, PROT_READ | PROT_WRITE,
                                       MAP_SHARED, drmfd, mreq.offset);
    if (vaddr == MAP_FAILED) { perror("[Display] mmap dumb"); goto cleanup_fb; }
    memset(vaddr, 0, creq.size);  /* 先清黑 */

    /* modeset: 把 fb 挂到 CRTC, 点亮 (进程持 master, 之后循环写 vaddr 即可见) */
    if (drmModeSetCrtc(drmfd, crtc_id, fb_id, 0, 0,
                       &conn->connector_id, 1, &mode) < 0) {
        perror("[Display] drmModeSetCrtc"); munmap(vaddr, creq.size); goto cleanup_fb;
    }
    printf("[Display] modeset OK — DRM 直显就绪\n");

    /* 4. 循环: 取去噪帧 → resize 直写 scanout buffer */
    int out_w = creq.width, out_h = creq.height;
    int frame_n = 0;
    while (p->running) {
        rgb_buf_t *rgb = (rgb_buf_t *)ring_get(&p->rgb_ring);
        if (!rgb) break;
        frame_n++;
        convert_to_fb(rgb->data, vaddr, out_w, out_h, dst_stride,
                      p->out_fmt == 0 ? OUT_NCHW : OUT_NHWC);
        if (frame_n == 1) printf("[Display] first frame shown\n");
        free(rgb->data);
        free(rgb);
    }

    munmap(vaddr, creq.size);
cleanup_fb:
    if (fb_id) drmModeRmFB(drmfd, fb_id);
cleanup_dumb:
    {
        struct drm_mode_destroy_dumb dreq;
        memset(&dreq, 0, sizeof(dreq));
        dreq.handle = creq.handle;
        drmIoctl(drmfd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
    }
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    drmDropMaster(drmfd);
    close(drmfd);

    system("/etc/init.d/S49weston start 2>/dev/null &");
    printf("[Display] stopped\n");
    return NULL;
}
