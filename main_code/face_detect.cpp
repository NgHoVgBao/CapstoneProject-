#include "face_detect.h"
#include "human_face_detect.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <cmath>

static const char *TAG = "FACE_DETECT";

extern "C" int face_detect_run(camera_fb_t *fb, face_box_t *out_boxes, int max_faces) {
    if (!fb || !fb->buf) {  
        ESP_LOGW(TAG, "Frame buffer rỗng!");
        return 0;
    }
    dl::image::img_t img;
    img.data = fb->buf;
    img.width = (uint16_t)fb->width;
    img.height = (uint16_t)fb->height;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565;
    static HumanFaceDetect *detect = new HumanFaceDetect();
    auto &results = detect->run(img);
    ESP_LOGI(TAG, "Số mặt tìm được: %d", (int)results.size());
    int count = 0;
    for (const auto &res : results) {
        if (count >= max_faces) break;
        out_boxes[count].x = res.box[0];
        out_boxes[count].y = res.box[1];
        out_boxes[count].w = res.box[2] - res.box[0];
        out_boxes[count].h = res.box[3] - res.box[1];
        out_boxes[count].score = res.score;
        
        for (int i = 0; i < 10; i++) {
            out_boxes[count].keypoint[i] = res.keypoint[i];
        }
        count++;
    }
    return count;
}

// Bỏ qua
extern "C" bool face_is_frontal(const face_box_t *face) {
    int left_eye_y  = face->keypoint[1];
    int right_eye_y = face->keypoint[7];
    int left_eye_x  = face->keypoint[0];
    int right_eye_x = face->keypoint[6];
    int chenh_lech_y = abs(left_eye_y - right_eye_y);
    int khoang_cach_hai_mat = abs(right_eye_x - left_eye_x);
    if (khoang_cach_hai_mat == 0) return false;
    float ty_le_lech = (float)chenh_lech_y / (float)khoang_cach_hai_mat;
    return ty_le_lech < 0.3f;
}