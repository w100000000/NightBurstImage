/*
 * fb_player.c — 播放预转换的 raw 帧到 /dev/fb0
 *
 * 配合 PC 端脚本使用:
 *   python3 prep_video.py input.mp4 frames/   # PC上转换
 *   scp -r frames/ root@<BOARD_IP>:/tmp/
 *   /tmp/fb_player /tmp/frames/ 30            # 30fps播放
 *
 * 编译 (板子上直接编译, 不需要交叉工具链):
 *   gcc -O2 -o fb_player fb_player.c -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/time.h>
#include <linux/fb.h>

#define FB_W  1024
#define FB_H  600
#define FB_BUF_SIZE (FB_W * FB_H * 4)  /* XRGB8888 */

static long long now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
}

/* sort frames numerically */
static int cmp(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <frame_dir> [fps=30] [loops=0]\n", argv[0]);
        return 1;
    }
    const char *dir   = argv[1];
    int fps           = argc > 2 ? atoi(argv[2]) : 30;
    int max_loops     = argc > 3 ? atoi(argv[3]) : 0;  /* 0=infinite */
    int frame_delay   = 1000000 / fps;

    /* collect frame files */
    DIR *d = opendir(dir);
    if (!d) { perror(dir); return 1; }

    char **files = NULL;
    int nfiles = 0;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        files = realloc(files, (nfiles + 1) * sizeof(char*));
        files[nfiles++] = strdup(ent->d_name);
    }
    closedir(d);
    if (nfiles == 0) { fprintf(stderr, "No files in %s\n", dir); return 1; }

    qsort(files, nfiles, sizeof(char*), cmp);
    printf("[Player] %d frames in %s, %d fps\n", nfiles, dir, fps);

    /* alloc buffers */
    unsigned char *fb_buf = malloc(FB_BUF_SIZE);
    unsigned char *frame  = malloc(FB_BUF_SIZE);

    /* stop weston */
    system("/etc/init.d/S49weston stop 2>/dev/null");
    usleep(500000);
    system("modetest -M rockchip -s 448@137:1024x600 2>/dev/null &");
    usleep(800000);

    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) { perror("/dev/fb0"); return 1; }
    printf("[Player] ready\n");

    int loop = 0;
    while (max_loops == 0 || loop < max_loops) {
        long long t0 = now_us();
        for (int i = 0; i < nfiles; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir, files[i]);

            FILE *f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "skip %s\n", path); continue; }
            size_t n = fread(frame, 1, FB_BUF_SIZE, f);
            fclose(f);
            if (n < FB_BUF_SIZE) {
                /* pad with black */
                memset(frame + n, 0, FB_BUF_SIZE - n);
            }

            /* write to fb */
            lseek(fb, 0, SEEK_SET);
            write(fb, frame, FB_BUF_SIZE);

            /* frame pacing */
            long long elapsed = now_us() - t0;
            long long target  = (long long)(i + 1) * frame_delay;
            if (elapsed < target) {
                usleep(target - elapsed);
            }
        }
        loop++;
        long long elapsed = now_us() - t0;
        printf("[Player] loop %d done, %.2f fps\n", loop,
               (double)nfiles / (elapsed / 1000000.0));
    }

    /* cleanup */
    for (int i = 0; i < nfiles; i++) free(files[i]);
    free(files); free(fb_buf); free(frame);
    close(fb);

    system("/etc/init.d/S49weston start 2>/dev/null &");
    printf("[Player] stopped\n");
    return 0;
}
