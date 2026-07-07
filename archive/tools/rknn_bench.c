/*
 * RKNN 模型性能测试 — 通用 benchmark
 *
 * 编译 (服务器交叉编译):
 *   $CC -O2 -o rknn_bench rknn_bench.c \
 *       -I${RKNN_API}/include -L${RKNN_API}/aarch64 -lrknnrt -lm
 *
 * 用法 (板子):
 *   ./rknn_bench <model.rknn> [warmup=5] [loops=20]
 *   ./rknn_bench y5fu_320x_sim.rknn
 *   ./rknn_bench y5fu_640x_sim.rknn
 *   ./rknn_bench rpv3_mdict_160_r3.rknn
 *   ./rknn_bench litemodel_cls_96x_r1.rknn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "rknn_api.h"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <model.rknn> [warmup=5] [loops=20]\n", argv[0]);
        return 1;
    }

    char *model_path = argv[1];
    int   warmup     = (argc >= 3) ? atoi(argv[2]) : 5;
    int   loops      = (argc >= 4) ? atoi(argv[3]) : 20;

    /* ── Load model ── */
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    printf("Model: %s\n", model_path);

    /* ── Query I/O ── */
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("  Inputs: %d, Outputs: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr in_attr[8];
    for (int i = 0; i < io_num.n_input; i++) {
        memset(&in_attr[i], 0, sizeof(rknn_tensor_attr));
        in_attr[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr[i], sizeof(rknn_tensor_attr));
        printf("  in[%d]:  dims=[%d,%d,%d,%d] size=%d type=%d fmt=%d scale=%f zp=%d\n",
               i, in_attr[i].dims[0], in_attr[i].dims[1],
               in_attr[i].dims[2], in_attr[i].dims[3],
               in_attr[i].size, in_attr[i].type, in_attr[i].fmt,
               in_attr[i].scale, in_attr[i].zp);
    }

    rknn_tensor_attr out_attr[8];
    for (int i = 0; i < io_num.n_output; i++) {
        memset(&out_attr[i], 0, sizeof(rknn_tensor_attr));
        out_attr[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr[i], sizeof(rknn_tensor_attr));
        printf("  out[%d]: dims=[%d,%d,%d,%d] size=%d type=%d fmt=%d\n",
               i, out_attr[i].dims[0], out_attr[i].dims[1],
               out_attr[i].dims[2], out_attr[i].dims[3],
               out_attr[i].size, out_attr[i].type, out_attr[i].fmt);
    }

    /* ── Allocate & fill random input ── */
    rknn_input  inputs[8];
    void        *input_bufs[8];
    memset(inputs, 0, sizeof(inputs));

    for (int i = 0; i < io_num.n_input; i++) {
        input_bufs[i] = malloc(in_attr[i].size);
        // fill with Gaussian-like random INT8 values
        for (int j = 0; j < in_attr[i].size; j++) {
            ((signed char *)input_bufs[i])[j] = (signed char)((rand() % 256) - 128);
        }
        inputs[i].index = i;
        inputs[i].type  = in_attr[i].type;  // INT8 or FP16...
        inputs[i].fmt   = in_attr[i].fmt;
        inputs[i].buf   = input_bufs[i];
        inputs[i].size  = in_attr[i].size;
    }

    /* ── Warmup ── */
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    for (int i = 0; i < warmup; i++) {
        rknn_run(ctx, NULL);
    }

    /* ── Benchmark ── */
    struct timeval t0, t1;
    double min_ms = 1e9, max_ms = 0, sum_ms = 0;

    for (int i = 0; i < loops; i++) {
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        gettimeofday(&t0, NULL);
        rknn_run(ctx, NULL);
        gettimeofday(&t1, NULL);

        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_usec - t0.tv_usec) / 1000.0;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
        sum_ms += ms;
    }

    double avg_ms = sum_ms / loops;
    printf("  Results (%d warmup + %d loops):\n", warmup, loops);
    printf("    min: %.2f ms  (%.1f FPS)\n", min_ms, 1000.0 / min_ms);
    printf("    avg: %.2f ms  (%.1f FPS)\n", avg_ms, 1000.0 / avg_ms);
    printf("    max: %.2f ms  (%.1f FPS)\n", max_ms, 1000.0 / max_ms);

    /* ── Cleanup ── */
    rknn_destroy(ctx);
    for (int i = 0; i < io_num.n_input; i++) {
        free(input_bufs[i]);
    }
    return 0;
}
