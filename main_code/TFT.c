#include "TFT.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_lcd_ili9341.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <string.h>
#define LCD_FLIP_Y 0
#define LCD_TEXT_FLIP_X 0

static const char *TAG_LCD = "ILI9341";
static esp_lcd_panel_handle_t panel_handle = NULL;
static DRAM_ATTR uint16_t header_buf[LCD_H_RES * 40];
static SemaphoreHandle_t color_done_sem = NULL;

static bool lcd_color_trans_done(esp_lcd_panel_io_handle_t panel_io,
                                 esp_lcd_panel_io_event_data_t *event_data,
                                 void *user_ctx) {
    BaseType_t high_task_woken = pdFALSE;
    xSemaphoreGiveFromISR((SemaphoreHandle_t)user_ctx, &high_task_woken);
    return high_task_woken == pdTRUE;
}

static esp_err_t lcd_draw_bitmap_sync(int x_start, int y_start, int x_end, int y_end,
                                      const void *data) {
    if (panel_handle == NULL || data == NULL) return ESP_ERR_INVALID_ARG;
    xSemaphoreTake(color_done_sem, 0);
    esp_err_t err = esp_lcd_panel_draw_bitmap(panel_handle, x_start, y_start,
                                              x_end, y_end, data);
    if (err != ESP_OK) return err;
    if (xSemaphoreTake(color_done_sem, pdMS_TO_TICKS(60)) != pdTRUE) {
        xSemaphoreTake(color_done_sem, 0);
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

static const uint8_t font8x16_basic[128][16] = {
    [' '] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['.'] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x60,0x60,0x00,0x00},
    [':'] = {0x00,0x00,0x00,0x00,0x00,0x60,0x60,0x00,0x00,0x60,0x60,0x00,0x00,0x00,0x00,0x00},
    ['/'] = {0x00,0x00,0x02,0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00,0x00,0x00,0x00,0x00,0x00},
    ['0'] = {0x00,0x00,0x3C,0x66,0x6E,0x76,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['1'] = {0x00,0x00,0x18,0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,0x00,0x00},
    ['2'] = {0x00,0x00,0x3C,0x66,0x06,0x0C,0x18,0x30,0x60,0x60,0x66,0x7E,0x00,0x00,0x00,0x00},
    ['3'] = {0x00,0x00,0x3C,0x66,0x06,0x06,0x1C,0x06,0x06,0x06,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['4'] = {0x00,0x00,0x0C,0x1C,0x3C,0x6C,0xCC,0xFE,0x0C,0x0C,0x0C,0x1E,0x00,0x00,0x00,0x00},
    ['5'] = {0x00,0x00,0x7E,0x60,0x60,0x78,0x0C,0x06,0x06,0x06,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['6'] = {0x00,0x00,0x3C,0x66,0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['7'] = {0x00,0x00,0xFE,0x66,0x06,0x0C,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00},
    ['8'] = {0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['9'] = {0x00,0x00,0x3C,0x66,0x66,0x66,0x66,0x3E,0x06,0x06,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['A'] = {0x00,0x00,0x10,0x38,0x6C,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['C'] = {0x00,0x00,0x3C,0x66,0xC2,0xC0,0xC0,0xC0,0xC0,0xC2,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['D'] = {0x00,0x00,0xF8,0x6C,0x66,0x66,0x66,0x66,0x66,0x66,0x6C,0xF8,0x00,0x00,0x00,0x00},
    ['E'] = {0x00,0x00,0xFE,0x62,0x62,0x68,0x78,0x68,0x60,0x62,0x62,0xFE,0x00,0x00,0x00,0x00},
    ['F'] = {0x00,0x00,0xFE,0x62,0x62,0x68,0x78,0x68,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    ['G'] = {0x00,0x00,0x3C,0x66,0xC2,0xC0,0xC0,0xDE,0xC6,0xC6,0x66,0x3A,0x00,0x00,0x00,0x00},
    ['H'] = {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['K'] = {0x00,0x00,0xE6,0x66,0x6C,0x68,0x70,0x68,0x6C,0x66,0x66,0xE6,0x00,0x00,0x00,0x00},
    ['I'] = {0x00,0x00,0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['M'] = {0x00,0x00,0xC6,0xEE,0xFE,0xFE,0xD6,0xC6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['N'] = {0x00,0x00,0xC6,0xE6,0xF6,0xFE,0xDE,0xCE,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00},
    ['O'] = {0x00,0x00,0x38,0x6C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x00,0x00,0x00,0x00},
    ['P'] = {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00},
    ['R'] = {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x6C,0x66,0x66,0x66,0xE6,0x00,0x00,0x00,0x00},
    ['S'] = {0x00,0x00,0x3C,0x66,0xC2,0xC0,0x70,0x1C,0x06,0x86,0x66,0x3C,0x00,0x00,0x00,0x00},
    ['T'] = {0x00,0x00,0xFE,0x92,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x38,0x00,0x00,0x00,0x00},
    ['U'] = {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00},
    ['Y'] = {0x00,0x00,0x66,0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00},
    ['L'] = {0x00,0x00,0xF0,0x60,0x60,0x60,0x60,0x60,0x62,0x62,0x66,0xFE,0x00,0x00,0x00,0x00},
    ['!'] = {0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00},
};

void lcd_ili9341_init(void) {
    ESP_LOGI(TAG_LCD, "Khoi tao TFT ILI9341...");

    color_done_sem = xSemaphoreCreateBinary();
    ESP_ERROR_CHECK(color_done_sem != NULL ? ESP_OK : ESP_ERR_NO_MEM);

    spi_bus_config_t buscfg = {
        .sclk_io_num = PIN_NUM_CLK,
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 240 * 40 * sizeof(uint16_t), // Khống chế Max Transfer vừa đủ
    };
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = PIN_NUM_DC,
        .cs_gpio_num = PIN_NUM_CS,
        .pclk_hz = 20 * 1000 * 1000, 
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .spi_mode = 0,
        .trans_queue_depth = 1,
    };
    
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_config, &io_handle));

    esp_lcd_panel_io_callbacks_t callbacks = {
        .on_color_trans_done = lcd_color_trans_done,
    };
    ESP_ERROR_CHECK(esp_lcd_panel_io_register_event_callbacks(io_handle, &callbacks, color_done_sem));

    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = PIN_NUM_RST,
        .rgb_endian = LCD_RGB_ENDIAN_BGR,
        .bits_per_pixel = 16,
    };
    
    ESP_ERROR_CHECK(esp_lcd_new_panel_ili9341(io_handle, &panel_config, &panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));

    if (PIN_NUM_BCKL >= 0) {
        gpio_set_direction(PIN_NUM_BCKL, GPIO_MODE_OUTPUT);
        gpio_set_level(PIN_NUM_BCKL, 1);
    }

    // Xóa màn hình an toàn
    lcd_fill_rect(0, 0, 240, 320, 0x0000);
}

void lcd_draw_bitmap_at(int x_start, int y_start, int x_end, int y_end, const uint16_t *data) {
    if (!panel_handle || !data) return;
    esp_err_t err = lcd_draw_bitmap_sync(x_start, y_start, x_end, y_end, data);
    if (err != ESP_OK) {
        ESP_LOGW("ILI9341", "Ve bitmap that bai: %s", esp_err_to_name(err));
    }
}

// TÔ KÍN KHỐI CHỮ NHẬT 
void lcd_fill_rect(int x, int y, int w, int h, uint16_t color) {
    if (!panel_handle || w <= 0 || h <= 0) return;

    // Chỉ cấp phát 1 buffer nhỏ 10 dòng (240x10) chuẩn MALLOC_CAP_DMA để vẽ cuốn chiếu
    int chunk_h = 10;
    size_t buf_size = w * chunk_h * sizeof(uint16_t);
    uint16_t *fill_buf = (uint16_t *)heap_caps_malloc(buf_size, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);

    if (fill_buf != NULL) {
        for (int i = 0; i < w * chunk_h; i++) {
            fill_buf[i] = color;
        }

        for (int cur_y = y; cur_y < y + h; cur_y += chunk_h) {
            int draw_h = (cur_y + chunk_h > y + h) ? (y + h - cur_y) : chunk_h;
            lcd_draw_bitmap_sync(x, cur_y, x + w, cur_y + draw_h, fill_buf);
        }
        free(fill_buf);
    }
}

// HÀM VẼ KHUNG VIỀN (DÙNG CHO BBOX NHẬN DIỆN KHUÔN MẶT)
void lcd_draw_rect(int x, int y, int w, int h, uint16_t color) {
    if (!panel_handle || w <= 0 || h <= 0) return;
    
    uint16_t line_buf[240];
    int max_w = (w > 240) ? 240 : w;
    for (int i = 0; i < max_w; i++) line_buf[i] = color;

    // Vẽ 4 cạnh viền
    lcd_draw_bitmap_sync(x, y, x + w, y + 1, line_buf);
    lcd_draw_bitmap_sync(x, y + h - 1, x + w, y + h, line_buf);
    lcd_draw_bitmap_sync(x, y, x + 1, y + h, line_buf);
    lcd_draw_bitmap_sync(x + w - 1, y, x + w, y + h, line_buf);
}

void lcd_display_enable(bool enable) {
    if (panel_handle == NULL) return;
    esp_lcd_panel_disp_on_off(panel_handle, enable);
    if (PIN_NUM_BCKL >= 0) {
        gpio_set_level(PIN_NUM_BCKL, enable ? 1 : 0);
    }
}

static inline uint8_t font_row(const uint8_t *bitmap, int row) {
#if LCD_FLIP_Y
    return bitmap[15 - row];   // bù lại việc phần cứng lật dọc
#else
    return bitmap[row];
#endif
}

static inline uint8_t font_col_mask(int col) {
#if LCD_TEXT_FLIP_X
    return 0x01 << col;     // đảo cột trong glyph
#else
    return 0x80 >> col;
#endif
}

void lcd_draw_char(int x, int y, char c, uint16_t color, uint16_t bg_color) {
    if (!panel_handle) return;
    
    // Bắt buộc cấp phát bộ nhớ DMA tĩnh hoặc từ MALLOC_CAP_DMA để tránh rác RAM khi vẽ SPI
    static DRAM_ATTR uint16_t char_buf[8 * 16];
    const uint8_t *bitmap = font8x16_basic[(uint8_t)c];

    for (int row = 0; row < 16; row++) {
    uint8_t bitmask = font_row(bitmap, row);   
    for (int col = 0; col < 8; col++) {
        char_buf[row * 8 + col] = (bitmask & font_col_mask(col)) ? color : bg_color;
        }
    }
    #if LCD_TEXT_FLIP_X
    x = LCD_H_RES - 8 - x;  
    #endif
    lcd_draw_bitmap_sync(x, y, x + 8, y + 16, char_buf);

}

void lcd_draw_string(int x, int y, const char *str, uint16_t color, uint16_t bg_color) {
    while (*str) {
        lcd_draw_char(x, y, *str, color, bg_color);
        x += 8;
        str++;
    }
}

// SỬA HÀM DƯỚI ĐÂY
void lcd_update_header(const char *text, uint16_t text_color, uint16_t bg_color) {
    if (!panel_handle || !text) return;

    static char last_text[32] = "";
    static uint16_t last_bg = 0xFFFF;

    if (strcmp(last_text, text) == 0 && last_bg == bg_color) {
        return;
    }

    strncpy(last_text, text, sizeof(last_text) - 1);
    last_bg = bg_color;

    for (int i = 0; i < LCD_H_RES * 40; i++) {
        header_buf[i] = bg_color;
    }

    // Tính tọa độ căn giữa
    int len = strlen(text);
    int text_width = len * 8;
    int start_x = (240 - text_width) / 2;
    if (start_x < 0) start_x = 0;
    int start_y = 12;

    // Vẽ từng ký tự trực tiếp vào buffer header, không gửi từng ký tự qua SPI.
    for (int char_index = 0; char_index < len; char_index++) {
        const uint8_t *bitmap = font8x16_basic[(uint8_t)text[char_index]];
        int char_x = start_x + char_index * 8;

        for (int row = 0; row < 16; row++) {
            uint8_t bitmask = font_row(bitmap, row);
            for (int col = 0; col < 8; col++) {
                int pixel_x = char_x + col;
                #if LCD_TEXT_FLIP_X
                pixel_x = LCD_H_RES - 1 - pixel_x;
                #endif
                int pixel_y = start_y + row;
                if (pixel_x >= 0 && pixel_x < LCD_H_RES && pixel_y >= 0 && pixel_y < 40) {
                    header_buf[pixel_y * LCD_H_RES + pixel_x] =
                        (bitmask & (0x80 >> col)) ? text_color : bg_color;
                }
            }
        }
    }

    esp_err_t err = lcd_draw_bitmap_sync(0, 0, LCD_H_RES, 40, header_buf);
    if (err != ESP_OK) {
        ESP_LOGW(TAG_LCD, "Ve header that bai: %s", esp_err_to_name(err));
    }
}