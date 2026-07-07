/*
 * NBINet RK3588 NPU Inference — 双输入 RAW 域三帧融合
 *
 * 输入: short_cat [1,544,960,8] NHWC INT8 + long_raw [1,544,960,4] NHWC INT8
 * 输出: RGB [1,3,1088,1920] NCHW FLOAT
 *
 * 编译 (服务器交叉编译):
 *   export CC=.../aarch64-linux-gcc
 *   $CC -O2 -o nbinet_infer nbinet_infer.c \
 *       -I${RKNN_API}/include -L${RKNN_API}/aarch64 -lrknnrt -lm
 *
 * 用法 (板子):
 *   ./nbinet_infer /tmp/nbinet.rknn short.bin long.bin output.rgb
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "rknn_api.h"

int main(int argc, char *argv[])
{
    if (argc < 5) {
        printf("Usage: %s <model.rknn> <short_cat.bin> <long_raw.bin> <output.rgb>\n", argv[0]);
        printf("  short_cat.bin: INT8 NHWC [1,544,960,8]  = 4,177,920 bytes\n");
        printf("  long_raw.bin:  INT8 NHWC [1,544,960,4]  = 2,088,960 bytes\n");
        printf("  output.rgb:    FLOAT NCHW [1,3,1088,1920] = 25,067,520 bytes\n");
        return 1;
    }

    char *model_path  = argv[1];
    char *short_path  = argv[2];
    char *long_path   = argv[3];
    char *output_path = argv[4];

    /* ── 1. Load RKNN model ── */
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    printf("[1/5] Model loaded: %s\n", model_path);

    /* ── 2. Query I/O info ── */
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("[2/5] Inputs=%d, Outputs=%d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr in_attr[2];
    for (int i = 0; i < 2; i++) {
        memset(&in_attr[i], 0, sizeof(rknn_tensor_attr));
        in_attr[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr[i], sizeof(rknn_tensor_attr));
        printf("      in[%d]: dims=[%d,%d,%d,%d] size=%d fmt=%d\n",
               i, in_attr[i].dims[0], in_attr[i].dims[1],
               in_attr[i].dims[2], in_attr[i].dims[3],
               in_attr[i].size, in_attr[i].fmt);
    }

    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    printf("      out[0]: dims=[%d,%d,%d,%d] size=%d\n",
           out_attr.dims[0], out_attr.dims[1],
           out_attr.dims[2], out_attr.dims[3], out_attr.size);

    /* ── 3. Load input data ── */
    unsigned char *short_buf = (unsigned char *)malloc(in_attr[0].size);
    unsigned char *long_buf  = (unsigned char *)malloc(in_attr[1].size);

    FILE *fp = fopen(short_path, "rb");
    if (!fp) { perror("fopen short"); return -1; }
    fread(short_buf, in_attr[0].size, 1, fp);
    fclose(fp);

    fp = fopen(long_path, "rb");
    if (!fp) { perror("fopen long"); return -1; }
    fread(long_buf, in_attr[1].size, 1, fp);
    fclose(fp);
    printf("[3/5] Input data loaded\n");

    /* ── 4. Run inference ── */
    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_INT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = short_buf;
    inputs[0].size  = in_attr[0].size;

    inputs[1].index = 1;
    inputs[1].type  = RKNN_TENSOR_INT8;
    inputs[1].fmt   = RKNN_TENSOR_NHWC;
    inputs[1].buf   = long_buf;
    inputs[1].size  = in_attr[1].size;

    rknn_inputs_set(ctx, 2, inputs);

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    ret = rknn_run(ctx, NULL);
    gettimeofday(&t1, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    long us = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("[4/5] Inference: %.2f ms (%.1f FPS)\n", us / 1000.0, 1000000.0 / us);

    /* ── 5. Get & save output ── */
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;  /* want float32 output */
    outputs[0].index = 0;
    rknn_outputs_get(ctx, 1, outputs, NULL);

    fp = fopen(output_path, "wb");
    fwrite(outputs[0].buf, outputs[0].size, 1, fp);
    fclose(fp);
    printf("[5/5] Output saved: %s (%d bytes)\n", output_path, outputs[0].size);

    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);
    free(short_buf);
    free(long_buf);

    return 0;
}
