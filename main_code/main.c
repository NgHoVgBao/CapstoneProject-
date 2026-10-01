#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/Task.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "face_detect.h"
#include "TFT.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "freertos/semphr.h"
#include "esp_websocket_client.h"
#include "freertos/event_groups.h"
#include "cJSON.h"


#define WIFI_SSID           "capheden"      
#define WIFI_PASS           "conheosuy"     
#define WEBSOCKET_SERVER_URI "ws://10.92.168.214:8000/ws/esp32" 

static const char *TAG = "DO AN 2";
static SemaphoreHandle_t lcd_mutex = NULL;
static esp_websocket_client_handle_t websocket_client = NULL;
static EventGroupHandle_t wifi_event_group = NULL;
TaskHandle_t xSR04Handle = NULL;
TaskHandle_t xButtonHandle = NULL;
TaskHandle_t xCameraHandle = NULL;
TaskHandle_t xSenderHandle = NULL;

#define WIFI_CONNECTED_BIT BIT0
volatile bool enable_ai = false; 
volatile bool enable_tft = false; 
static volatile int64_t manual_wake_until = 0;

#define IMAGE_SEND_MODE_REGISTER   0x01
#define IMAGE_SEND_MODE_ATTENDANCE 0x02

// Structure dùng để truyền pointer FrameBuffer sang Sender Task
typedef struct {
    uint8_t mode;
    camera_fb_t *fb;
} img_msg_t;

#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_Y9      16
#define CAM_PIN_Y8      17
#define CAM_PIN_Y7      18
#define CAM_PIN_Y6      12
#define CAM_PIN_Y5      10
#define CAM_PIN_Y4      8
#define CAM_PIN_Y3      9
#define CAM_PIN_Y2      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

#define TRIG_PIN  GPIO_NUM_48
#define ECHO_PIN  GPIO_NUM_19
#define BTN_PIN   GPIO_NUM_20
// 48, 19 (SR04), 20 (SW), 21, 47, 48, 45, 41 (TFT)

typedef enum {
    MODE_DIEM_DANH = 0,
    MODE_DANG_KY
} system_mode_t;

volatile system_mode_t g_system_mode = MODE_DIEM_DANH;

static int reg_capture_count = 1;     
static bool reg_completed = false;    
static int64_t reg_delay_until = 0;   

typedef enum {
    ATTEND_IDLE = 0,
    ATTEND_WAIT_RESPONSE,
    ATTEND_SHOW_RESULT
} attend_state_t;

static volatile attend_state_t g_attend_state = ATTEND_IDLE;
static volatile int64_t g_attend_timeout_until = 0;
static volatile int64_t g_attend_reset_until = 0;
static char g_received_id[32] = {0};
static volatile bool g_is_success = false;

// --- HÀM BẬT TẮT MÀN TFT ---
static void set_display_state(bool enable) {
    if (enable_tft != enable) {
        if (xSemaphoreTake(lcd_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
            enable_tft = enable;
            enable_ai = enable;
            lcd_display_enable(enable);
            xSemaphoreGive(lcd_mutex);
            ESP_LOGI(TAG, "Màn hình: %s", enable ? "BAT" : "TAT");
        } else {
            ESP_LOGW(TAG, "Không lấy được Mutex để %s màn hình!", enable ? "BAT" : "TAT");
        }
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
        ESP_LOGW(TAG, "Wi-Fi mat ket noi, dang ket noi lai...");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
        ESP_LOGI(TAG, "Da nhan IP: "IPSTR, IP2STR(&event->ip_info.ip));
    }
}

static void wifi_init_sta(void) {
    wifi_event_group = xEventGroupCreate();
    if (wifi_event_group == NULL) return;

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *wifi_netif = esp_netif_create_default_wifi_sta();
    if (wifi_netif == NULL) return;

    wifi_init_config_t wifi_init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init_config));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_connect());

    EventBits_t bits = xEventGroupWaitBits(
        wifi_event_group,
        WIFI_CONNECTED_BIT,
        pdFALSE,
        pdFALSE,
        pdMS_TO_TICKS(5000) 
    );  

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Ket noi Wi-Fi thanh cong, san sang khoi tao WebSocket.");
    } else {
        ESP_LOGE(TAG, "Ket noi Wi-Fi pride/timeout thoi gian cho!");
    }
}

// Truyền dữ liệu trực tiếp struct chứa pointer fb
static void request_image_send(uint8_t mode, camera_fb_t *fb) {
    if (xSenderHandle != NULL && fb != NULL) {
        img_msg_t *msg = malloc(sizeof(img_msg_t));
        if (msg) {
            msg->mode = mode;
            msg->fb = fb;
            if (xTaskNotify(xSenderHandle, (uint32_t)msg, eSetValueWithOverwrite) != pdPASS) {
                free(msg);
            }
        }
    }
}

static void image_sender_task(void *pvParameters) {
    uint32_t notification_value;

    while (1) {
        if (xTaskNotifyWait(0, UINT32_MAX, &notification_value, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        img_msg_t *msg = (img_msg_t *)notification_value;
        if (msg == NULL) continue;

        camera_fb_t *fb = msg->fb;
        uint8_t requested_mode = msg->mode;
        free(msg); // Giải phóng struct bọc

        if (fb == NULL) continue;

        if (websocket_client == NULL || !esp_websocket_client_is_connected(websocket_client)) {
            ESP_LOGW(TAG, "Khong gui anh: WebSocket chua ket noi");
            esp_camera_fb_return(fb); // Trả buffer chuẩn xác
            continue;
        }

        size_t packet_len = fb->len + 1;
        uint8_t *packet = heap_caps_malloc(packet_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (packet == NULL) {
            esp_camera_fb_return(fb);
            continue;
        }

        packet[0] = requested_mode;
        memcpy(packet + 1, fb->buf, fb->len);
        
        // Trả buffer camera ngay sau khi copy xong bộ nhớ
        esp_camera_fb_return(fb);

        int sent = esp_websocket_client_send_bin(
            websocket_client, (const char *)packet, packet_len, pdMS_TO_TICKS(5000));
        
        if (sent >= 0 && requested_mode == IMAGE_SEND_MODE_ATTENDANCE) {
            g_attend_state = ATTEND_WAIT_RESPONSE;
            g_attend_timeout_until = esp_timer_get_time() + 5000000; // 5 giây timeout
        }

        free(packet);
    }
}

static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    if (event_id == WEBSOCKET_EVENT_DATA) {
        if (data->data_ptr != NULL && data->data_len > 0 && g_attend_state == ATTEND_WAIT_RESPONSE) {
            char *json_str = malloc(data->data_len + 1);
            if (json_str) {
                memcpy(json_str, data->data_ptr, data->data_len);
                json_str[data->data_len] = '\0';

                cJSON *root = cJSON_Parse(json_str);
                if (root) {
                    cJSON *matched = cJSON_GetObjectItem(root, "is_matched");
                    cJSON *user_id = cJSON_GetObjectItem(root, "user_id_num");

                    if (cJSON_IsBool(matched) && cJSON_IsTrue(matched) && cJSON_IsString(user_id)) {
                        strncpy(g_received_id, user_id->valuestring, sizeof(g_received_id) - 1);
                        g_is_success = true;
                    } else {
                        g_is_success = false;
                    }
                    cJSON_Delete(root);

                    g_attend_timeout_until = 0; // Xóa cờ timeout
                    g_attend_state = ATTEND_SHOW_RESULT;
                    g_attend_reset_until = esp_timer_get_time() + 2000000; // Giữ màn hình kết quả 2 giây
                }
                free(json_str);
            }
        }
    }
}

static void websocket_init(void) {
    esp_websocket_client_config_t websocket_config = {
        .uri = WEBSOCKET_SERVER_URI,
    };
    websocket_client = esp_websocket_client_init(&websocket_config);
    if (websocket_client == NULL) return;

    esp_websocket_register_events(websocket_client, WEBSOCKET_EVENT_DATA, websocket_event_handler, (void *)websocket_client);

    esp_err_t err = esp_websocket_client_start(websocket_client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Khong ket noi duoc WebSocket: %s", esp_err_to_name(err));
    }
}

bool init_camera(void) {
    camera_config_t config;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_d7 = CAM_PIN_Y9;
    config.pin_d6 = CAM_PIN_Y8;
    config.pin_d5 = CAM_PIN_Y7;
    config.pin_d4 = CAM_PIN_Y6;
    config.pin_d3 = CAM_PIN_Y5;
    config.pin_d2 = CAM_PIN_Y4;
    config.pin_d1 = CAM_PIN_Y3;
    config.pin_d0 = CAM_PIN_Y2;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_pclk = CAM_PIN_PCLK;
    
    config.xclk_freq_hz = 20000000; 
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    
    config.pixel_format = PIXFORMAT_RGB565; 
    config.frame_size = FRAMESIZE_240X240;  
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) return false;
    
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_framesize(s, FRAMESIZE_240X240);
        s->set_vflip(s, 0);   
        s->set_hmirror(s, 0); 
    }
    return true;
}

void init_button(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BTN_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);
}

void button_task(void *pvParameters) {
    int last_state = 1;
    while (1) {
        int current_state = gpio_get_level(BTN_PIN);
        if (last_state == 1 && current_state == 0) {
            vTaskDelay(pdMS_TO_TICKS(50));
            if (gpio_get_level(BTN_PIN) == 0) {
                if (g_system_mode == MODE_DIEM_DANH) {
                    g_system_mode = MODE_DANG_KY;
                    reg_capture_count = 1;
                    reg_completed = false;
                    reg_delay_until = esp_timer_get_time() + 1000000; // Delay 1s khi vừa bấm nút chuyển mode
                    manual_wake_until = esp_timer_get_time() + 10000000;
                    set_display_state(true);
                    ESP_LOGW(TAG, "🔘 Nút bấm: Chuyển sang ---> MODE_DANG_KY");
                } else {
                    g_system_mode = MODE_DIEM_DANH;
                    reg_completed = false;
                    reg_capture_count = 1;
                    manual_wake_until = 0;
                    g_attend_state = ATTEND_IDLE;
                    ESP_LOGI(TAG, "🔘 Nút bấm: Chuyển sang ---> MODE_DIEM_DANH");
                }
                int timeout = 0;
                while (gpio_get_level(BTN_PIN) == 0 && timeout < 100) {
                    vTaskDelay(pdMS_TO_TICKS(20));
                    timeout++;
                }
                current_state = 1;
            }
        }       
        last_state = current_state;
        vTaskDelay(pdMS_TO_TICKS(30));
    }
}

void camera_ai_task(void *pvParameters) {
    face_box_t faces[5];
    int offset_x = 0;
    int offset_y = 40;
    
    int64_t face_detect_start_time = 0; // Mốc thời gian bắt đầu thấy mặt

    while (1) {
        if (!enable_tft) {
            face_detect_start_time = 0;
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            int so_mat = face_detect_run(fb, faces, 5);
            int64_t now = esp_timer_get_time();

            // Tính thời gian duy trì mặt trước ống kính
            if (so_mat >= 1) {
                if (face_detect_start_time == 0) {
                    face_detect_start_time = now;
                }
            } else {
                face_detect_start_time = 0; // Reset nếu bỏ mặt ra khỏi khung hình
            }

            bool face_held_1s = (face_detect_start_time != 0) && ((now - face_detect_start_time) >= 1000000);

            // --- XỬ LÝ CHUYỂN TRẠNG THÁI & LOGIC ---
            if (g_system_mode == MODE_DANG_KY) {
                if (reg_completed) {
                    if (now >= reg_delay_until) {
                        g_system_mode = MODE_DIEM_DANH;
                        reg_completed = false;
                        reg_capture_count = 1;
                        manual_wake_until = 0;
                        set_display_state(false);
                        esp_camera_fb_return(fb);
                        continue;
                    }
                } else {
                    // Cần duy trì mặt đủ 1 giây + hết thời gian chờ reg_delay_until
                    if (face_held_1s && now >= reg_delay_until) {
                        if (reg_capture_count < 3) {
                            reg_capture_count++;
                            reg_delay_until = now + 1500000; // Đợi 1.5s giữa các lần chụp
                            face_detect_start_time = 0; // Reset để bắt đếm lại 1s cho tấm tiếp theo
                        } else {
                            reg_completed = true;
                            reg_delay_until = now + 2000000;
                            request_image_send(IMAGE_SEND_MODE_REGISTER, fb);
                            fb = NULL; // Chuyển quyền quản lý fb cho image_sender_task
                        }
                    }
                }
            } else {
                // Mode Điểm Danh
                switch (g_attend_state) {
                    case ATTEND_IDLE:
                        // Yêu cầu giữ mặt ổn định đúng 1s mới gửi ảnh
                        if (face_held_1s && now < manual_wake_until) {
                            g_attend_state = ATTEND_WAIT_RESPONSE;
                            g_attend_timeout_until = now + 5000000;
                            request_image_send(IMAGE_SEND_MODE_ATTENDANCE, fb);
                            fb = NULL; // Chuyển quyền quản lý fb cho image_sender_task
                            face_detect_start_time = 0;
                        }
                        break;

                    case ATTEND_WAIT_RESPONSE:
                        if (now >= g_attend_timeout_until) {
                            g_is_success = false;
                            g_attend_state = ATTEND_SHOW_RESULT;
                            g_attend_reset_until = now + 2000000;
                        }
                        break;

                    case ATTEND_SHOW_RESULT:
                        if (now >= g_attend_reset_until) {
                            g_attend_state = ATTEND_IDLE;
                            g_received_id[0] = '\0';
                            face_detect_start_time = 0;
                            set_display_state(false); 
                            if (fb) esp_camera_fb_return(fb);
                            continue;
                        }
                        break;
                }
            }

            // --- RENDER DỮ LIỆU LÊN LCD ---
            if (fb && xSemaphoreTake(lcd_mutex, pdMS_TO_TICKS(30))) {
                
                // 1. Cập nhật Header
                if (g_system_mode == MODE_DANG_KY) {
                    if (reg_completed) {
                        lcd_update_header("REGIS COMPLETED !", 0xFFFF, 0xF800);
                    } else {
                        char msg[21];
                        snprintf(msg, sizeof(msg), "REGISTER: CAP %d/3", reg_capture_count);
                        lcd_update_header(msg, 0xFFFF, 0xF800);
                    }
                } else {
                    switch (g_attend_state) {
                        case ATTEND_IDLE:
                            lcd_update_header("MODE: ATTENDANCE", 0xFFFF, 0x07E0);
                            break;
                        case ATTEND_WAIT_RESPONSE:
                            lcd_update_header("CHECKING...", 0xFFFF, 0xFFE0);
                            break;
                        case ATTEND_SHOW_RESULT:
                            if (g_is_success) {
                                char msg[48];
                                snprintf(msg, sizeof(msg), "SUCCESS: %s", g_received_id);
                                lcd_update_header(msg, 0xFFFF, 0x07E0);
                            } else {
                                lcd_update_header("FAILED", 0xFFFF, 0xF800);
                            }
                            break;
                    }
                }

                // 2. Vẽ Bitmap Camera
                lcd_draw_bitmap_at(offset_x, offset_y, offset_x + fb->width,
                                   offset_y + fb->height, (uint16_t *)fb->buf);

                // 3. Vẽ khung BBox nhận diện mặt
                for (int i = 0; i < so_mat; i++) {
                    int tft_x = faces[i].x + offset_x;
                    int tft_y = faces[i].y + offset_y;
                    int cam_w = faces[i].w;
                    int cam_h = faces[i].h;
                    if (tft_x < 0) tft_x = 0;
                    if (tft_y < 0) tft_y = 0;
                    if (tft_x + cam_w > 240) cam_w = 240 - tft_x;
                    if (tft_y + cam_h > 280) cam_h = 280 - tft_y;

                    uint16_t box_color = (g_system_mode == MODE_DANG_KY) ? 0xF800 : 0x07E0;
                    lcd_draw_rect(tft_x, tft_y, cam_w, cam_h, box_color);
                }

                xSemaphoreGive(lcd_mutex);
            }

            if (fb) {
                esp_camera_fb_return(fb);
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void init_ultrasonic(void) {
    gpio_config_t io_conf_trig = {
        .pin_bit_mask = (1ULL << TRIG_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf_trig);

    gpio_config_t io_conf_echo = {
        .pin_bit_mask = (1ULL << ECHO_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf_echo);

    vTaskDelay(pdMS_TO_TICKS(50));
    gpio_set_level(TRIG_PIN, 0);
}

float get_distance_cm(void) {
    gpio_set_level(TRIG_PIN, 0);
    esp_rom_delay_us(2);
    gpio_set_level(TRIG_PIN, 1);
    esp_rom_delay_us(10);
    gpio_set_level(TRIG_PIN, 0);

    int wait_time = 0;
    while (gpio_get_level(ECHO_PIN) == 0) {
        esp_rom_delay_us(2);
        wait_time += 2;
        if (wait_time > 2000) return -1.0f; 
    }
    int64_t echo_start = esp_timer_get_time();

    wait_time = 0;
    while (gpio_get_level(ECHO_PIN) == 1) {
        esp_rom_delay_us(2);
        wait_time += 2;
        if (wait_time > 12000) return -1.0f; 
    }
    int64_t echo_end = esp_timer_get_time();

    int64_t duration = echo_end - echo_start;
    if (duration <= 0) return -1.0f;

    return (duration * 0.0343f) / 2.0f;
}

void ultrasonic_task(void *pvParameters) {
    while (1) {
        float distance = get_distance_cm();
        int64_t now = esp_timer_get_time();
        if (distance > 0.0f && distance <= 30.0f) {
            manual_wake_until = now + 5000000;  
            set_display_state(true);
        } 
        else if (distance > 35.0f || distance < 0.0f) {
            if (g_system_mode == MODE_DIEM_DANH && g_attend_state == ATTEND_IDLE) {
                set_display_state(false);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(60));
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void app_main(void)
{
    lcd_mutex = xSemaphoreCreateMutex();
    esp_err_t nvs_result = nvs_flash_init();
    if (nvs_result == ESP_ERR_NVS_NO_FREE_PAGES || nvs_result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_result = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_result);
    
    lcd_ili9341_init();
    init_button();
    init_ultrasonic();
    
    if (xSemaphoreTake(lcd_mutex, portMAX_DELAY)) {
        lcd_fill_rect(0, 0, 240, 320, 0x0000); 
        lcd_display_enable(false); 
        xSemaphoreGive(lcd_mutex);
    }

    wifi_init_sta();
    websocket_init();

    xTaskCreatePinnedToCore(ultrasonic_task, "ultrasonic_task", 3072, NULL, 4, &xSR04Handle, 0);
    xTaskCreatePinnedToCore(button_task, "button_task", 4096, NULL, 3, &xButtonHandle, 0);
    xTaskCreatePinnedToCore(image_sender_task, "image_sender_task", 4096, NULL, 4, &xSenderHandle, 1);
    
    if (init_camera()) {
        xTaskCreatePinnedToCore(camera_ai_task, "camera_ai_task", 32 * 2048, NULL, 5, &xCameraHandle, 1);
    }
}

#ifdef __cplusplus
}
#endif