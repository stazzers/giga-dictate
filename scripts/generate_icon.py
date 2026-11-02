import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_all_icons():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(root, "giga_dictate", "resources", "source_drawing.jpg")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source drawing not found: {source_path}")

    # 1. Load and extract drawing strokes with transparent alpha
    src = Image.open(source_path).convert("RGBA")
    arr = np.array(src)
    diff = 255 - arr[:, :, :3]
    dist = np.max(diff, axis=2).astype(np.float32)
    # Extract crisp strokes while eliminating paper texture background
    alpha = np.clip((dist - 14) * (255.0 / (dist.max() - 14)), 0, 255).astype(np.uint8)

    drawing = Image.new("RGBA", src.size, (0, 0, 0, 0))
    drawing_arr = np.array(drawing)
    drawing_arr[:, :, :3] = arr[:, :, :3]
    drawing_arr[:, :, 3] = alpha
    drawing = Image.fromarray(drawing_arr)

    bbox = drawing.getbbox()
    cropped = drawing.crop(bbox)

    # 2. Build 1024x1024 App Icon in Apple Light Tones (Porcelain White to Silver Frost)
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    margin = int(size * 0.082)
    radius = int(size * 0.225)
    sq_bbox = [margin, margin, size - margin, size - margin]

    # Authentic macOS Squircle Drop Shadow
    shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_layer)
    s_bbox = [margin, margin + int(size * 0.022), size - margin, size - margin + int(size * 0.022)]
    s_draw.rounded_rectangle(s_bbox, radius=radius, fill=(0, 0, 0, 55))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(int(size * 0.028)))
    img.alpha_composite(shadow_layer)

    # Squircle Light Background Gradient: Pure Porcelain White -> Silky Platinum Silver
    bg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    for y in range(margin, size - margin):
        t = (y - margin) / float(size - 2 * margin)
        # Subtle light gradient: #FFFFFF (255, 255, 255) -> #EAF0F6 (234, 240, 246)
        r = int(255 * (1 - t) + 234 * t)
        g = int(255 * (1 - t) + 240 * t)
        b = int(255 * (1 - t) + 246 * t)
        bg_draw.line([(margin, y), (size - margin, y)], fill=(r, g, b, 255))

    # Ambient soft center radiance for depth and warmth
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    cx, cy = size // 2, size // 2
    glow_r = int(size * 0.38)
    glow_draw.ellipse([cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r], fill=(255, 120, 100, 22))
    glow = glow.filter(ImageFilter.GaussianBlur(int(size * 0.09)))
    bg.alpha_composite(glow)

    # Cut to squircle shape
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle(sq_bbox, radius=radius, fill=255)
    bg.putalpha(mask)
    img.alpha_composite(bg)

    # Outer border (delicate slate rim so the white squircle stands out clearly on white backgrounds)
    rim = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rim_draw = ImageDraw.Draw(rim)
    rim_draw.rounded_rectangle(sq_bbox, radius=radius, outline=(160, 175, 195, 140), width=int(size * 0.005))
    # Top inner glass highlight (pure white sheen)
    rim_draw.rounded_rectangle(
        [margin + 2, margin + 2, size - margin - 2, size - margin - 2],
        radius=radius - 2,
        outline=(255, 255, 255, 230),
        width=int(size * 0.008),
    )
    img.alpha_composite(rim)

    # Scale user drawing to fit elegantly inside squircle (height ~640px)
    target_h = int(size * 0.64)
    target_w = int(cropped.width * (target_h / cropped.height))
    scaled = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    pos_x = (size - target_w) // 2
    pos_y = (size - target_h) // 2

    # Tactile drop shadow beneath the hand-drawn strokes
    drawing_shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    s_alpha = scaled.split()[3]
    s_mask = Image.new("RGBA", scaled.size, (50, 20, 20, 60))
    s_mask.putalpha(s_alpha)
    drawing_shadow.paste(s_mask, (pos_x, pos_y + int(size * 0.012)))
    drawing_shadow = drawing_shadow.filter(ImageFilter.GaussianBlur(int(size * 0.010)))
    img.alpha_composite(drawing_shadow)

    # Composite the user's authentic hand-drawn logo
    img.alpha_composite(scaled, (pos_x, pos_y))

    app_icon_png = os.path.join(root, "app_icon_1024.png")
    img.save(app_icon_png)
    print(f"Saved light-theme icon to {app_icon_png}")

    # 3. Generate AppIcon.icns
    iconset_dir = os.path.join(root, "AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)
    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for s, name in sizes:
        resized = img.resize((s, s), Image.Resampling.LANCZOS)
        resized.save(os.path.join(iconset_dir, name))

    icns_path = os.path.join(root, "AppIcon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
    subprocess.run(["rm", "-rf", iconset_dir], check=True)
    print(f"Generated {icns_path}")

    # Copy into GigaDictate.app and /Applications/GigaDictate.app
    from AppKit import NSImage, NSWorkspace
    app_icon_ns = NSImage.alloc().initWithContentsOfFile_(icns_path)

    for app_path in [
        os.path.join(root, "GigaDictate.app"),
        "/Applications/GigaDictate.app",
    ]:
        if os.path.exists(app_path):
            app_res = os.path.join(app_path, "Contents", "Resources")
            os.makedirs(app_res, exist_ok=True)
            subprocess.run(["cp", icns_path, os.path.join(app_res, "AppIcon.icns")], check=True)
            NSWorkspace.sharedWorkspace().setIcon_forFile_options_(app_icon_ns, app_path, 0)
            subprocess.run(["touch", app_path], check=True)
            print(f"Applied white-toned icon to {app_path}")

    lsreg = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
    if os.path.exists(lsreg) and os.path.exists("/Applications/GigaDictate.app"):
        subprocess.run([lsreg, "-f", "/Applications/GigaDictate.app"], check=False)
        print("Refreshed LaunchServices registration")

    # 4. Generate Menu Bar icons (template black + recording red)
    res_dir = os.path.join(root, "giga_dictate", "resources")
    os.makedirs(res_dir, exist_ok=True)

    # Black template icon for macOS Menu Bar
    crop_alpha = cropped.split()[3]
    black_arr = np.zeros((cropped.height, cropped.width, 4), dtype=np.uint8)
    black_arr[:, :, 3] = np.array(crop_alpha)
    black_src = Image.fromarray(black_arr)

    h_2x = 36
    w_2x = int(cropped.width * (h_2x / cropped.height))
    h_1x = 18
    w_1x = int(cropped.width * (h_1x / cropped.height))

    black_src.resize((w_1x, h_1x), Image.Resampling.LANCZOS).save(os.path.join(res_dir, "menu_icon.png"))
    black_src.resize((w_2x, h_2x), Image.Resampling.LANCZOS).save(os.path.join(res_dir, "menu_icon@2x.png"))

    # Red recording icon for active state
    red_arr = np.zeros((cropped.height, cropped.width, 4), dtype=np.uint8)
    red_arr[:, :, 0] = 255
    red_arr[:, :, 1] = 59
    red_arr[:, :, 2] = 48
    red_arr[:, :, 3] = np.array(crop_alpha)
    red_src = Image.fromarray(red_arr)

    red_src.resize((w_1x, h_1x), Image.Resampling.LANCZOS).save(os.path.join(res_dir, "menu_icon_rec.png"))
    red_src.resize((w_2x, h_2x), Image.Resampling.LANCZOS).save(os.path.join(res_dir, "menu_icon_rec@2x.png"))
    print("Menu bar icons generated in giga_dictate/resources/")


if __name__ == "__main__":
    generate_all_icons()
