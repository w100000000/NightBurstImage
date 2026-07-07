/* ring_buf.c */
#include "pipeline_types.h"
#include <stdlib.h>
#include <string.h>

int ring_init(ring_buf_t *rb, int size, size_t elem_size)
{
    rb->size = size;
    rb->elem_size = elem_size;
    rb->head = 0; rb->tail = 0; rb->quit = false;
    pthread_mutex_init(&rb->lock, NULL);
    pthread_cond_init(&rb->not_empty, NULL);
    pthread_cond_init(&rb->not_full, NULL);
    rb->slots = calloc(size, sizeof(void *));
    for (int i = 0; i < size; i++)
        rb->slots[i] = malloc(elem_size);
    return (rb->slots != NULL) ? 0 : -1;
}

void ring_destroy(ring_buf_t *rb)
{
    for (int i = 0; i < rb->size; i++) free(rb->slots[i]);
    free(rb->slots);
    pthread_mutex_destroy(&rb->lock);
    pthread_cond_destroy(&rb->not_empty);
    pthread_cond_destroy(&rb->not_full);
}

int ring_put(ring_buf_t *rb, void *elem)
{
    pthread_mutex_lock(&rb->lock);
    int next = (rb->head + 1) % rb->size;
    while (next == rb->tail && !rb->quit)
        pthread_cond_wait(&rb->not_full, &rb->lock);
    if (rb->quit) { pthread_mutex_unlock(&rb->lock); return -1; }
    memcpy(rb->slots[rb->head], elem, rb->elem_size);
    rb->head = next;
    pthread_cond_signal(&rb->not_empty);
    pthread_mutex_unlock(&rb->lock);
    return 0;
}

void* ring_get(ring_buf_t *rb)
{
    pthread_mutex_lock(&rb->lock);
    while (rb->head == rb->tail && !rb->quit)
        pthread_cond_wait(&rb->not_empty, &rb->lock);
    if (rb->quit) { pthread_mutex_unlock(&rb->lock); return NULL; }
    void *result;
    memcpy(&result, rb->slots[rb->tail], rb->elem_size); /* 读回拷贝 */
    rb->tail = (rb->tail + 1) % rb->size;
    pthread_cond_signal(&rb->not_full);
    pthread_mutex_unlock(&rb->lock);
    return result;
}

int ring_try_put(ring_buf_t *rb, void *elem)
{
    pthread_mutex_lock(&rb->lock);
    int next = (rb->head + 1) % rb->size;
    if (next == rb->tail) { pthread_mutex_unlock(&rb->lock); return -1; }
    memcpy(rb->slots[rb->head], elem, rb->elem_size);
    rb->head = next;
    pthread_cond_signal(&rb->not_empty);
    pthread_mutex_unlock(&rb->lock);
    return 0;
}

void* ring_try_get(ring_buf_t *rb)
{
    pthread_mutex_lock(&rb->lock);
    if (rb->head == rb->tail) { pthread_mutex_unlock(&rb->lock); return NULL; }
    void *result;
    memcpy(&result, rb->slots[rb->tail], rb->elem_size);
    rb->tail = (rb->tail + 1) % rb->size;
    pthread_cond_signal(&rb->not_full);
    pthread_mutex_unlock(&rb->lock);
    return result;
}

void ring_stop(ring_buf_t *rb)
{
    pthread_mutex_lock(&rb->lock);
    rb->quit = true;
    pthread_cond_broadcast(&rb->not_empty);
    pthread_cond_broadcast(&rb->not_full);
    pthread_mutex_unlock(&rb->lock);
}
