import screeninfo

def get_screen_info():
    # Get second monitor details
    monitors = screeninfo.get_monitors()

    # Two monitor setup
    if len(monitors) > 1:
        second_monitor = monitors[1]
        x_offset, y_offset = second_monitor.x, second_monitor.y
        screen_width, screen_height = second_monitor.width, second_monitor.height

    # Default to primary monitor
    else:
        x_offset, y_offset = 0, 0
        screen_width, screen_height = monitors[0].width, monitors[0].height 

    # 95% screen width and height
    window_width = int(screen_width * 0.95)
    window_height = int(screen_height * 0.95)

    # Center window
    window_x = x_offset + (screen_width - window_width) // 2
    window_y = y_offset + (screen_height - window_height) // 2

    return window_width, window_height, window_x, window_y