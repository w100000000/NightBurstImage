/*
 * main.c — ASI585MC 实时去噪 + LPR 管线
 *
 * 用法:
 *   ./pipeline \
 *     --model  /mnt/sdcard/nbinet_272x480.rknn \
 *     --lpr    /mnt/sdcard \
 *     --output /mnt/sdcard/plates \
 *     --exp-short 16667 \
 *     --exp-long  83333
 */
#include "pipeline_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

static pipeline_t g_pipe;

static void on_signal(int sig)
{
    (void)sig;
    g_pipe.running = false;
    ring_stop(&g_pipe.raw_ring);
    ring_stop(&g_pipe.rgb_ring);
    ring_stop(&g_pipe.lpr_ring);
}

int main(int argc, char *argv[])
{
    memset(&g_pipe, 0, sizeof(g_pipe));
    snprintf(g_pipe.model_path, sizeof(g_pipe.model_path),
             "/mnt/sdcard/nbinet_272x480.rknn");
    snprintf(g_pipe.lpr_model_dir, sizeof(g_pipe.lpr_model_dir),
             "/mnt/sdcard");
    snprintf(g_pipe.storage_dir, sizeof(g_pipe.storage_dir),
             "/mnt/sdcard/plates");
    g_pipe.expo_short_us = 16667;
    g_pipe.expo_long_us  = 83333;
    g_pipe.running       = true;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1 < argc)
            snprintf(g_pipe.model_path, sizeof(g_pipe.model_path), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--lpr") && i+1 < argc)
            snprintf(g_pipe.lpr_model_dir, sizeof(g_pipe.lpr_model_dir), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc)
            snprintf(g_pipe.storage_dir, sizeof(g_pipe.storage_dir), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--exp-short") && i+1 < argc)
            g_pipe.expo_short_us = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exp-long") && i+1 < argc)
            g_pipe.expo_long_us = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-lpr"))
            g_pipe.no_lpr = true;
        else if (!strcmp(argv[i], "--no-display"))
            g_pipe.no_display = true;
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [options]\n"
                   "  --model PATH      NPU model (default: /mnt/sdcard/nbinet_272x480.rknn)\n"
                   "  --lpr   PATH      LPR model dir (default: /mnt/sdcard)\n"
                   "  --output PATH     plate results dir (default: /mnt/sdcard/plates)\n"
                   "  --exp-short US    short exposure us (default: 16667)\n"
                   "  --exp-long  US    long exposure us (default: 83333)\n"
                   "  --no-lpr          disable LPR\n"
                   "  --no-display      disable MIPI display (test USB only)\n",
                   argv[0]);
            return 0;
        }
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("=== ELF2 ASI585MC Pipeline ===\n");
    printf("NPU model    : %s\n", g_pipe.model_path);
    printf("LPR models   : %s\n", g_pipe.lpr_model_dir);
    printf("Storage      : %s\n", g_pipe.storage_dir);
    printf("Exposure     : short=%dus  long=%dus\n",
           g_pipe.expo_short_us, g_pipe.expo_long_us);

    ring_init(&g_pipe.raw_ring, RING_SIZE, sizeof(raw_buf_t *));
    ring_init(&g_pipe.rgb_ring, RING_SIZE, sizeof(rgb_buf_t *));
    ring_init(&g_pipe.lpr_ring, RING_SIZE, sizeof(rgb_buf_t *));

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

#ifdef __cplusplus
    extern "C" {
#endif
    extern void *capture_thread(void *);
    extern void *npu_thread(void *);
    extern void *display_thread(void *);
    extern void *lpr_thread(void *);
#ifdef __cplusplus
    }
#endif

    printf("\nStarting threads...\n");
    pthread_create(&g_pipe.cap_thread, NULL, capture_thread, &g_pipe);
    pthread_create(&g_pipe.npu_thread, NULL, npu_thread, &g_pipe);
    if (!g_pipe.no_display)
        pthread_create(&g_pipe.dsp_thread, NULL, display_thread, &g_pipe);
    if (!g_pipe.no_lpr)
        pthread_create(&g_pipe.lpr_thread, NULL, lpr_thread, &g_pipe);

    while (g_pipe.running) {
        sleep(2);
        printf("[STAT] frames=%lld drops=%lld npu=%.1fms lpr=%.1fms\n",
               (long long)g_pipe.frame_cnt,
               (long long)g_pipe.drop_cnt,
               g_pipe.npu_ms, g_pipe.lpr_ms);
    }

    printf("\nShutting down...\n");
    pthread_join(g_pipe.cap_thread, NULL);
    pthread_join(g_pipe.npu_thread, NULL);
    if (!g_pipe.no_display)
        pthread_join(g_pipe.dsp_thread, NULL);
    if (!g_pipe.no_lpr)
        pthread_join(g_pipe.lpr_thread, NULL);

    ring_destroy(&g_pipe.raw_ring);
    ring_destroy(&g_pipe.rgb_ring);
    ring_destroy(&g_pipe.lpr_ring);
    printf("Pipeline stopped. Total frames: %lld\n", (long long)g_pipe.frame_cnt);
    return 0;
}
