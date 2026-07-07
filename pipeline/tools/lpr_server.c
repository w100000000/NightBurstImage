/*
 * lpr_server.c — 无依赖 HTTP 服务器, 手机/小程序查询车牌
 *
 *   GET /            → 网页: 按车牌归组, 可搜索, 每辆车显示全部抓拍图+时间+帧数
 *   GET /api/plates  → JSON (按车牌归组)
 *   GET /files/<名>  → 返回 plates 目录下文件 (jpg/txt)
 *
 * 数据源 <dir>/plates.txt, 每行形如:
 *   frame=63 ts=1234 plate=苏A6D8F8 type=蓝牌(0) conf=0.74 img=cap_63.jpg yolo_ms=.. lpr_ms=..
 *
 * 编译: aarch64-buildroot-linux-gnu-gcc -O2 -o lpr_server lpr_server.c
 * 运行: ./lpr_server 8080 /mnt/sdcard/plates
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/stat.h>

static char g_dir[512] = "/mnt/sdcard/plates";

#define MAX_GROUP   128     /* 最多不同车牌 */
#define MAX_SHOT     64     /* 每辆车最多保留抓拍数 */

typedef struct {
    char frame[16], conf[16], img[128], ts[32];
} shot_t;
typedef struct {
    char plate[64];
    int  count;             /* 命中总帧数 */
    char best_conf[16];
    int  last_seq;          /* 最后出现的行号, 用于排序 */
    shot_t shots[MAX_SHOT];
    int  nshots;
} group_t;

static void send_resp(int fd, const char *status, const char *ctype,
                      const char *body, long blen)
{
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %ld\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        status, ctype, blen);
    write(fd, hdr, n);
    if (body && blen > 0) write(fd, body, blen);
}

static int get_field(const char *line, const char *key, char *buf, int buflen)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=", key);
    const char *p = strstr(line, pat);
    if (!p) return 0;
    p += strlen(pat);
    int i = 0;
    while (*p && *p != ' ' && *p != '\n' && *p != '\r' && i < buflen - 1)
        buf[i++] = *p++;
    buf[i] = 0;
    return 1;
}

/* 读 plates.txt → 按车牌归组 */
static int load_groups(group_t *g)
{
    char path[600];
    snprintf(path, sizeof(path), "%s/plates.txt", g_dir);
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    int ng = 0, seq = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char plate[64] = "", conf[16] = "", img[128] = "", frame[16] = "", ts[32] = "";
        if (!get_field(line, "plate", plate, sizeof(plate)) || !plate[0]) continue;
        get_field(line, "conf", conf, sizeof(conf));
        get_field(line, "img",  img,  sizeof(img));
        get_field(line, "frame", frame, sizeof(frame));
        get_field(line, "ts",   ts,   sizeof(ts));
        seq++;

        /* 找已有车牌 */
        int gi = -1;
        for (int i = 0; i < ng; i++)
            if (strcmp(g[i].plate, plate) == 0) { gi = i; break; }
        if (gi < 0) {
            if (ng >= MAX_GROUP) continue;
            gi = ng++;
            strncpy(g[gi].plate, plate, 63);
            g[gi].count = 0; g[gi].nshots = 0;
            strcpy(g[gi].best_conf, "0");
        }
        g[gi].count++;
        g[gi].last_seq = seq;
        if (atof(conf) > atof(g[gi].best_conf)) strncpy(g[gi].best_conf, conf, 15);
        /* 有图才存为一次抓拍 (环形保留最近 MAX_SHOT 张) */
        if (img[0]) {
            int si = g[gi].nshots % MAX_SHOT;
            strncpy(g[gi].shots[si].frame, frame, 15);
            strncpy(g[gi].shots[si].conf,  conf, 15);
            strncpy(g[gi].shots[si].img,   img, 127);
            strncpy(g[gi].shots[si].ts,    ts, 31);
            g[gi].nshots++;
        }
    }
    fclose(f);
    return ng;
}

static void handle_api(int fd)
{
    static group_t g[MAX_GROUP];
    int ng = load_groups(g);

    /* 按 last_seq 降序 (最新的车牌在前), 简单选择排序 */
    int idx[MAX_GROUP];
    for (int i = 0; i < ng; i++) idx[i] = i;
    for (int i = 0; i < ng; i++)
        for (int j = i + 1; j < ng; j++)
            if (g[idx[j]].last_seq > g[idx[i]].last_seq) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }

    char *json = (char *)malloc(1 << 20);   /* 1MB */
    int jl = 0;
    jl += sprintf(json + jl, "[");
    for (int k = 0; k < ng; k++) {
        group_t *gp = &g[idx[k]];
        if (k) jl += sprintf(json + jl, ",");
        jl += sprintf(json + jl,
            "{\"plate\":\"%s\",\"count\":%d,\"best_conf\":\"%s\",\"shots\":[",
            gp->plate, gp->count, gp->best_conf);
        int tot = gp->nshots < MAX_SHOT ? gp->nshots : MAX_SHOT;
        for (int s = 0; s < tot; s++) {
            if (s) jl += sprintf(json + jl, ",");
            jl += sprintf(json + jl,
                "{\"frame\":\"%s\",\"conf\":\"%s\",\"img\":\"%s\",\"ts\":\"%s\"}",
                gp->shots[s].frame, gp->shots[s].conf, gp->shots[s].img, gp->shots[s].ts);
        }
        jl += sprintf(json + jl, "]}");
    }
    jl += sprintf(json + jl, "]");
    send_resp(fd, "200 OK", "application/json; charset=utf-8", json, jl);
    free(json);
}

static void handle_file(int fd, const char *name)
{
    if (strstr(name, "..")) { send_resp(fd, "403 Forbidden", "text/plain", "no", 2); return; }
    char path[700];
    snprintf(path, sizeof(path), "%s/%s", g_dir, name);
    int ffd = open(path, O_RDONLY);
    if (ffd < 0) { send_resp(fd, "404 Not Found", "text/plain", "not found", 9); return; }
    struct stat st; fstat(ffd, &st);
    const char *ctype = "application/octet-stream";
    if (strstr(name, ".jpg") || strstr(name, ".jpeg")) ctype = "image/jpeg";
    else if (strstr(name, ".bmp")) ctype = "image/bmp";
    else if (strstr(name, ".txt")) ctype = "text/plain; charset=utf-8";
    char hdr[256];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %ld\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        ctype, (long)st.st_size);
    write(fd, hdr, n);
    char buf[8192]; ssize_t r;
    while ((r = read(ffd, buf, sizeof(buf))) > 0) write(fd, buf, r);
    close(ffd);
}

static const char *HTML_PAGE =
"<!DOCTYPE html><html><head><meta charset=utf-8>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<title>车牌识别查询</title><style>"
"*{box-sizing:border-box}body{font-family:sans-serif;margin:0;background:#f0f2f5;padding-bottom:20px}"
"h2{background:#1677ff;color:#fff;margin:0;padding:14px;font-size:18px}"
"#s{width:92%;margin:12px 4%;padding:10px 14px;border:1px solid #ccc;border-radius:20px;font-size:15px}"
".card{background:#fff;margin:10px 4%;border-radius:10px;padding:12px;box-shadow:0 1px 4px #0001}"
".pl{font-size:20px;font-weight:bold;color:#1677ff}"
".meta{color:#888;font-size:13px;margin:4px 0 8px}"
".shots{display:flex;overflow-x:auto;gap:8px;padding-bottom:4px}"
".shot{flex:0 0 auto;text-align:center}"
".shot img{height:90px;border-radius:6px;border:1px solid #eee;display:block}"
".shot span{font-size:11px;color:#999}"
".empty{padding:50px;text-align:center;color:#999}"
"</style></head><body><h2>车牌识别查询 <span id=cnt></span></h2>"
"<input id=s placeholder='搜索车牌号, 如 苏A 或 6D8'>"
"<div id=list></div><div id=e class=empty></div>"
"<script>"
"var DATA=[];"
"function render(){"
"var q=document.getElementById('s').value.trim().toUpperCase();"
"var d=DATA.filter(p=>!q||p.plate.toUpperCase().indexOf(q)>=0);"
"var L=document.getElementById('list'),e=document.getElementById('e');L.innerHTML='';"
"if(!d.length){e.textContent=DATA.length?'无匹配车牌':'暂无记录';return}e.textContent='';"
"d.forEach(p=>{var sh=p.shots.map(s=>"
"'<div class=shot><img src=/files/'+s.img+' onclick=\"window.open(this.src)\">"
"<span>帧'+s.frame+' ('+s.conf+')</span></div>').join('');"
"if(!sh)sh='<span style=color:#bbb>(无抓拍图)</span>';"
"L.innerHTML+='<div class=card><div class=pl>'+p.plate+'</div>"
"<div class=meta>命中 '+p.count+' 帧 · 最高置信度 '+p.best_conf+' · 抓拍 '+p.shots.length+' 张</div>"
"<div class=shots>'+sh+'</div></div>'})}"
"function load(){fetch('/api/plates').then(r=>r.json()).then(d=>{DATA=d;"
"document.getElementById('cnt').textContent='('+d.length+'辆)';render()})}"
"document.getElementById('s').oninput=render;"
"load();setInterval(load,3000);"
"</script></body></html>";

int main(int argc, char *argv[])
{
    int port = (argc > 1) ? atoi(argv[1]) : 8080;
    if (argc > 2) snprintf(g_dir, sizeof(g_dir), "%s", argv[2]);
    signal(SIGPIPE, SIG_IGN);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    listen(srv, 8);
    printf("[lpr_server] listening on :%d  dir=%s\n", port, g_dir);

    while (1) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) continue;
        char req[1024] = {0};
        read(fd, req, sizeof(req) - 1);
        char method[8], path[512];
        if (sscanf(req, "%7s %511s", method, path) != 2) { close(fd); continue; }

        if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0)
            send_resp(fd, "200 OK", "text/html; charset=utf-8", HTML_PAGE, (long)strlen(HTML_PAGE));
        else if (strcmp(path, "/api/plates") == 0)
            handle_api(fd);
        else if (strncmp(path, "/files/", 7) == 0)
            handle_file(fd, path + 7);
        else
            send_resp(fd, "404 Not Found", "text/plain", "404", 3);
        close(fd);
    }
    return 0;
}
