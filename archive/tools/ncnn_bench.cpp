/*
 * ncnn CPU Benchmark — 支持多输入模型
 *
 * 编译 (服务器):
 *   $CXX -O2 -o ncnn_bench ncnn_bench.cpp \
 *        -I<NCNN_ROOT>/src -I<NCNN_ROOT>/build/src \
 *        -L<NCNN_ROOT>/build/src -lncnn -lpthread
 *
 * 单输入用法:
 *   ./ncnn_bench /mnt/sdcard/y5fu_320x_sim       320 320 3
 *   ./ncnn_bench /mnt/sdcard/litemodel_cls_96x_r1  96  96 3
 *
 * 多输入用法 (NBINet):
 *   ./ncnn_bench /mnt/sdcard/nbinet_272x480    272 480 8   272 480 4
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include "net.h"

struct Shape { int h, w, c; };

int main(int argc, char *argv[])
{
    if (argc < 5) {
        printf("Usage: %s <model_prefix> <h1> <w1> <c1> [h2 w2 c2 ...] [warmup=5] [loops=20]\n", argv[0]);
        printf("  NBINet: %s nbinet_272x480 272 480 8 272 480 4\n", argv[0]);
        printf("  YOLO:   %s y5fu_320x_sim 320 320 3\n", argv[0]);
        return 1;
    }

    std::string prefix = argv[1];
    std::string param  = prefix + ".ncnn.param";
    std::string bin    = prefix + ".ncnn.bin";

    /* Parse shapes: remaining args are (h w c)*, last 2 may be warmup/loops */
    std::vector<Shape> shapes;
    int i = 2;
    while (i + 2 < argc) {
        Shape s;
        s.h = atoi(argv[i]);
        s.w = atoi(argv[i+1]);
        s.c = atoi(argv[i+2]);
        shapes.push_back(s);
        i += 3;
    }
    int warmup = (i < argc) ? atoi(argv[i]) : 5;
    int loops  = (i+1 < argc) ? atoi(argv[i+1]) : 20;

    /* ── Load model ── */
    ncnn::Net net;
    int ret = net.load_param(param.c_str());
    if (ret != 0) { printf("load_param %s fail! ret=%d\n", param.c_str(), ret); return -1; }
    ret = net.load_model(bin.c_str());
    if (ret != 0) { printf("load_model %s fail! ret=%d\n", bin.c_str(), ret); return -1; }

    printf("Model: %s\n", prefix.c_str());

    const auto &in_names  = net.input_names();
    const auto &out_names = net.output_names();
    printf("  Inputs: %zu  Outputs: %zu\n", in_names.size(), out_names.size());
    for (size_t j = 0; j < in_names.size(); j++)
        printf("  in[%zu]: %s\n", j, in_names[j]);
    for (size_t j = 0; j < out_names.size(); j++)
        printf("  out[%zu]: %s\n", j, out_names[j]);

    /* ── Create random input ── */
    srand(42);
    std::vector<ncnn::Mat> inputs;
    for (int j = 0; j < (int)shapes.size(); j++) {
        ncnn::Mat m(shapes[j].w, shapes[j].h, shapes[j].c);
        for (int k = 0; k < m.total(); k++)
            m[k] = (rand() % 65536) / 65536.0f;
        inputs.push_back(m);
    }

    /* ── Warmup ── */
    for (int j = 0; j < warmup; j++) {
        ncnn::Extractor ex = net.create_extractor();
        for (int k = 0; k < (int)shapes.size(); k++)
            ex.input(in_names[k], inputs[k]);
        ncnn::Mat out;
        ex.extract(out_names[0], out);
    }

    /* ── Benchmark ── */
    struct timeval t0, t1;
    double min_ms = 1e9, max_ms = 0, sum_ms = 0;

    for (int j = 0; j < loops; j++) {
        ncnn::Extractor ex = net.create_extractor();
        for (int k = 0; k < (int)shapes.size(); k++)
            ex.input(in_names[k], inputs[k]);

        gettimeofday(&t0, NULL);
        ncnn::Mat out;
        ex.extract(out_names[0], out);
        gettimeofday(&t1, NULL);

        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_usec - t0.tv_usec) / 1000.0;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
        sum_ms += ms;
    }

    printf("  Results (%d warmup + %d loops):\n", warmup, loops);
    printf("    min: %.2f ms  (%.1f FPS)\n", min_ms, 1000.0 / min_ms);
    printf("    avg: %.2f ms  (%.1f FPS)\n", sum_ms / loops, 1000.0 * loops / sum_ms);
    printf("    max: %.2f ms  (%.1f FPS)\n", max_ms, 1000.0 / max_ms);

    return 0;
}
