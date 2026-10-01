idf_component_register(
    SRCS "TFT.c" "main.c" "face_detect.cpp" 
    INCLUDE_DIRS "." 
    REQUIRES 
    esp_http_server
    espressif__esp32-camera 
    esp_http_client
    espressif__esp-dl 
    espressif__human_face_detect 
    esp_websocket_client
    esp_wifi 
    esp_timer 
    nvs_flash 
    esp_event 
    driver 
    esp_lcd
    json
)