/*
 * npu_to_fb.c — NPU 输出 → MIPI 屏幕直写
 *
 * 编译 (交叉编译，服务器):
 *   $CC -O2 -o npu_to_fb npu_to_fb.c -lm
 *
 * 板子:
 *   /etc/init.d/S49weston stop
 *   ./npu_to_fb /tmp/output.rgb  [--rotate 1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fb.h>

#define NPU_W    544
#define NPU_H    960
#define FB_W     1024
#define FB_H     600

/* Bilinear resize + convert float32 NCHW → uint8 XRGB8888 */
static void convert_frame(const float *src, /* NCHW [3][NPU_H][NPU_W] */
                          uint32_t *dst, int rotate)
{
    int ow, oh;
    if (rotate) { ow = FB_H; oh = FB_W; }  /* 旋转后: resize到 600×1024 → 输出到 1024×600 */
    else        { ow = FB_W; oh = FB_H; }

    float scale_x = (float)NPU_W / ow;
    float scale_y = (float)NPU_H / oh;

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            float sx = (x + 0.5f) * scale_x - 0.5f;
            float sy = (y + 0.5f) * scale_y - 0.5f;
            if (sx < 0) sx = 0; if (sx > NPU_W - 1) sx = NPU_W - 1;
            if (sy < 0) sy = 0; if (sy > NPU_H - 1) sy = NPU_H - 1;

            int x0 = (int)sx, y0 = (int)sy;
            int x1 = x0 + 1; if (x1 >= NPU_W) x1 = x0;
            int y1 = y0 + 1; if (y1 >= NPU_H) y1 = y0;

            float fx = sx - x0, fy = sy - y0;
            float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;

            const float *c0 = src;              /* R */
            const float *c1 = src + NPU_H * NPU_W;   /* G */
            const float *c2 = src + 2 * NPU_H * NPU_W; /* B */

#define SAMPLE(c, px, py) ((c)[(py) * NPU_W + (px)])

            float r = (1-fy)*((1-fx)*SAMPLE(c0,x0,y0) + fx*SAMPLE(c0,x1,y0)) +
                           fy*((1-fx)*SAMPLE(c0,x0,y1) + fx*SAMPLE(c0,x1,y1));
            float g = (1-fy)*((1-fx)*SAMPLE(c1,x0,y0) + fx*SAMPLE(c1,x1,y0)) +
                           fy*((1-fx)*SAMPLE(c1,x0,y1) + fx*SAMPLE(c1,x1,y1));
            float b = (1-fy)*((1-fx)*SAMPLE(c2,x0,y0) + fx*SAMPLE(c2,x1,y0)) +
                           fy*((1-fx)*SAMPLE(c2,x0,y1) + fx*SAMPLE(c2,x1,y1));
#undef SAMPLE

            int rb = 0, gb = 0, bb = 0;
            if (r > 1.0f) r = 1.0f; if (r < 0) r = 0;
            if (g > 1.0f) g = 1.0f; if (g < 0) g = 0;
            if (b > 1.0f) b = 1.0f; if (b < 0) b = 0;
            rb = (int)(r * 255.0f + 0.5f);
            gb = (int)(g * 255.0f + 0.5f);
            bb = (int)(b * 255.0f + 0.5f);

            uint32_t px = (uint32_t)bb | ((uint32_t)gb << 8) |
                          ((uint32_t)rb << 16); /* BGRX */

            if (rotate)
                dst[x * oh + (oh - 1 - y)] = px;  /* 旋转 + 翻转 */
            else
                dst[y * ow + x] = px;
        }
    }
}

int main(int argc, char *argv[])
{
    const char *input_path  = "/tmp/output.rgb";
    const char *fb_path     = "/dev/fb0";
    int         rotate      = 1;  /* NPU竖屏→MIPI横屏 */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rotate")) rotate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--input")) input_path = argv[++i];
        else if (!strcmp(argv[i], "--fb")) fb_path = argv[++i];
    }

    /* ── Read NPU output (float32 NCHW) ── */
    size_t nbytes = 3 * NPU_W * NPU_H * sizeof(float);
    float *src = (float *)malloc(nbytes);
    FILE *fp = fopen(input_path, "rb");
    if (!fp) { perror(input_path); return 1; }
    fread(src, 1, nbytes, fp);
    fclose(fp);
    printf("Read %zu bytes from %s\n", nbytes, input_path);

    /* ── Allocate framebuffer ── */
    size_t  fb_size = FB_W * FB_H * sizeof(uint32_t);
    uint32_t *fb    = (uint32_t *)calloc(1, fb_size);

    /* ── Convert ── */
    convert_frame(src, fb, rotate);

    /* ── Open framebuffer ── */
    int fbfd = open(fb_path, O_RDWR);
    if (fbfd < 0) { perror(fb_path); return 1; }

    struct fb_var_screeninfo vinfo;
    ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo);
    printf("fb: %dx%d, bpp=%d\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);

    /* ── Write ── */
    ssize_t w = write(fbfd, fb, fb_size);
    if (w < 0) perror("write fb");
    else       printf("Wrote %zd bytes to %s\n", w, fb_path);

    close(fbfd);
    free(src);
    free(fb);
    return 0;
}
