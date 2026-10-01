#pragma once
#include "esp_camera.h"
#include <stdbool.h>    
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x, y, w, h;     // toa do bounding box cua khuon mat
    float score;        // do tin cay model tim thay day la khuon mat
    // 5 diem landmark, thu tu dung theo model tra ve:
    // [0-1]=left_eye, [2-3]=left_mouth, [4-5]=nose, [6-7]=right_eye, [8-9]=right_mouth
    int keypoint[10];
} face_box_t;

int face_detect_run(camera_fb_t *fb, face_box_t *out_boxes, int max_faces);

// Kiem tra khuon mat co "chinh dien" hay khong, dua vao vi tri 2 mat + mui
// Tra ve true neu 2 mat gan nhu ngang hang nhau (it nghieng dau)
bool face_is_frontal(const face_box_t *face);

#ifdef __cplusplus
}
#endif
