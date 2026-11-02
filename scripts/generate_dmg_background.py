import os
from PIL import Image, ImageDraw, ImageFont

def generate_background(output_path: str):
    # Width and height in points: 560 x 360
    # Retina scale 2x: 1120 x 720
    w = 1120
    h = 720
    img = Image.new("RGBA", (w, h), (242, 243, 247, 255))
    draw = ImageDraw.Draw(img)

    # 1. Subtle soft gradient / backdrop
    for y in range(h):
        ratio = y / float(h)
        # Soft subtle silver/white gradient
        r = int(246 - ratio * 10)
        g = int(247 - ratio * 10)
        b = int(251 - ratio * 8)
        draw.line([(0, y), (w, y)], fill=(r, g, b, 255))

    # 2. Card container panel (subtle rounded glass card)
    card_margin_x = 40
    card_margin_y = 40
    card_rect = [card_margin_x, card_margin_y, w - card_margin_x, h - card_margin_y]
    draw.rounded_rectangle(card_rect, radius=24, fill=(255, 255, 255, 200), outline=(225, 228, 235, 255), width=2)

    # 3. Typography
    font_bold_path = "/System/Library/Fonts/HelveticaNeue.ttc"
    try:
        font_title = ImageFont.truetype(font_bold_path, size=40, index=1) # Bold
        font_sub = ImageFont.truetype(font_bold_path, size=24, index=0)   # Regular
        font_hint = ImageFont.truetype(font_bold_path, size=20, index=0)  # Regular
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_hint = ImageFont.load_default()

    title_text = "GigaDictate"
    sub_text = "Перетащите приложение в папку «Программы» для установки"

    # Draw Title
    t_bbox = draw.textbbox((0, 0), title_text, font=font_title)
    t_w = t_bbox[2] - t_bbox[0]
    draw.text(((w - t_w) // 2, 65), title_text, font=font_title, fill=(30, 32, 40, 255))

    # Draw Subtitle
    s_bbox = draw.textbbox((0, 0), sub_text, font=font_sub)
    s_w = s_bbox[2] - s_bbox[0]
    draw.text(((w - s_w) // 2, 120), sub_text, font=font_sub, fill=(100, 105, 120, 255))

    # 4. Sleek Arrow between App (left ~280 in 2x) and Applications (right ~840 in 2x)
    # Center is at x=560, y=360
    arrow_center_x = w // 2
    arrow_center_y = 350
    arrow_len = 100
    arrow_color = (0, 122, 255, 220) # Apple Blue

    # Draw horizontal arrow body
    line_y = arrow_center_y
    x_start = arrow_center_x - arrow_len // 2
    x_end = arrow_center_x + arrow_len // 2
    draw.line([(x_start, line_y), (x_end, line_y)], fill=arrow_color, width=8)

    # Draw arrow head
    head_size = 20
    draw.polygon([
        (x_end + 10, line_y),
        (x_end - head_size, line_y - head_size),
        (x_end - head_size + 4, line_y),
        (x_end - head_size, line_y + head_size),
    ], fill=arrow_color)

    # Bottom footer / hint
    hint_text = "Apple Silicon Native (M1 / M2 / M3 / M4) • Локальное распознавание речи"
    h_bbox = draw.textbbox((0, 0), hint_text, font=font_hint)
    h_w = h_bbox[2] - h_bbox[0]
    draw.text(((w - h_w) // 2, h - 85), hint_text, font=font_hint, fill=(140, 145, 160, 255))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Background image generated: {output_path} ({w}x{h})")

if __name__ == "__main__":
    generate_background("build/background.png")
