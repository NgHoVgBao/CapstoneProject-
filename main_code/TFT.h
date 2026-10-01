#pragma once
#include <stdint.h>
#include <stdbool.h>

#define LCD_HOST        SPI2_HOST
#define PIN_NUM_MISO    -1          
#define PIN_NUM_MOSI    47          
#define PIN_NUM_CLK     21          
#define PIN_NUM_CS      42          
#define PIN_NUM_DC      45          
#define PIN_NUM_RST     41          
#define PIN_NUM_BCKL    14          

#define LCD_H_RES       240
#define LCD_V_RES       320

#define LCD_COLOR_BLACK   0x0000
#define LCD_COLOR_WHITE   0xFFFF
#define LCD_COLOR_RED     0xF800
#define LCD_COLOR_GREEN   0x07E0
#define LCD_COLOR_BLUE    0x001F

void lcd_ili9341_init(void);
void lcd_display_enable(bool enable);
void lcd_draw_rect(int x, int y, int w, int h, uint16_t color);
void lcd_fill_rect(int x, int y, int w, int h, uint16_t color);
void lcd_draw_bitmap_at(int x_start, int y_start, int x_end, int y_end, const uint16_t *data);
void lcd_draw_char(int x, int y, char c, uint16_t color, uint16_t bg_color);
void lcd_draw_string(int x, int y, const char *str, uint16_t color, uint16_t bg_color);
void lcd_update_header(const char *text, uint16_t text_color, uint16_t bg_color);