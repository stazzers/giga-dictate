import logging
import math

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSCompositingOperationClear,
    NSEvent,
    NSFont,
    NSFontAttributeName,
    NSGradient,
    NSGraphicsContext,
    NSLineBreakByTruncatingHead,
    NSLineBreakByWordWrapping,
    NSPanel,
    NSPoint,
    NSPointInRect,
    NSRect,
    NSRectFillUsingOperation,
    NSScreen,
    NSShadow,
    NSSize,
    NSString,
    NSStringDrawingUsesFontLeading,
    NSStringDrawingUsesLineFragmentOrigin,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSTextField,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWorkspace,
)
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCopyParameterizedAttributeValue,
    AXUIElementCreateApplication,
    AXValueGetValue,
    kAXBoundsForRangeParameterizedAttribute,
    kAXFocusedUIElementAttribute,
    kAXPositionAttribute,
    kAXSelectedTextRangeAttribute,
    kAXSizeAttribute,
    kAXValueCGPointType,
    kAXValueCGRectType,
    kAXValueCGSizeType,
)
from Foundation import NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer

logger = logging.getLogger("giga_dictate.hud")

# Borderless, non-activating floating panel (never steals focus)
STYLE_MASK = NSWindowStyleMaskBorderless | (1 << 7)
FLOATING_LEVEL = 1000  # Floating above windows

CLASSIC_WIDTH = 72.0
CLASSIC_HEIGHT = 52.0
PREVIEW_WIDTH = 290.0
PREVIEW_HEIGHT = 46.0
PREVIEW_MIN_WIDTH = 200.0
PREVIEW_MAX_WIDTH = 540.0

NUM_BARS = 5
CENTER_WEIGHTS = tuple(
    math.sin(((i + 0.5) / float(NUM_BARS)) * math.pi) ** 0.82 for i in range(NUM_BARS)
)


def locate_caret_or_cursor(
    hud_w: float = 72.0,
    hud_h: float = 52.0,
    screens: list | None = None,
    target_pt: NSPoint | None = None,
) -> NSPoint:
    """
    Finds screen coordinates (in Cocoa coordinate space) for:
    1. The blinking text cursor (caret) of the active input field (Safari, Notes, Word, Telegram, etc.)
    2. Fallback: Focused element bounds
    3. Fallback: Current mouse cursor position
    4. Fallback: Bottom-center of the screen
    Accurately resolves multi-screen geometries and avoids cross-display clamping.
    """
    screen = NSScreen.mainScreen()
    screens_list = screens if screens is not None else NSScreen.screens()
    primary_h = screens_list[0].frame().size.height if screens_list else 900.0
    active_screen_frame = (
        screens_list[0].visibleFrame()
        if (screens is not None and screens_list)
        else (screen.visibleFrame() if screen else NSRect(NSPoint(0, 0), NSSize(1440, 900)))
    )

    target_x = target_pt.x if target_pt is not None else None
    target_y = target_pt.y if target_pt is not None else None
    target_screen_frame = active_screen_frame
    caret_y = None
    caret_h = None

    if target_pt is None:
        try:
            front_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if front_app is not None:
                pid = front_app.processIdentifier()
                app_ax = AXUIElementCreateApplication(pid)
                err, focused = AXUIElementCopyAttributeValue(
                    app_ax, kAXFocusedUIElementAttribute, None
                )
                if err == 0 and focused is not None:
                    # 1. Exact blinking text cursor (caret) range bounds
                    err_r, sel_r = AXUIElementCopyAttributeValue(
                        focused, kAXSelectedTextRangeAttribute, None
                    )
                    if err_r == 0 and sel_r is not None:
                        err_b, b_val = AXUIElementCopyParameterizedAttributeValue(
                            focused, kAXBoundsForRangeParameterizedAttribute, sel_r, None
                        )
                        if err_b == 0 and b_val is not None:
                            ok, rect = AXValueGetValue(b_val, kAXValueCGRectType, None)
                            if ok and rect.size.height > 0:
                                # Convert AX coordinates (top-left origin) to Cocoa (bottom-left origin)
                                caret_x = rect.origin.x
                                caret_y = primary_h - rect.origin.y - rect.size.height
                                caret_h = rect.size.height

                                # Place HUD directly underneath the blinking caret
                                target_x = caret_x + (rect.size.width / 2.0) - (hud_w / 2.0)
                                target_y = caret_y - hud_h - 6.0

                    # 2. Fallback: Focused element position and size
                    if target_x is None:
                        err_p, pos_val = AXUIElementCopyAttributeValue(
                            focused, kAXPositionAttribute, None
                        )
                        err_s, sz_val = AXUIElementCopyAttributeValue(
                            focused, kAXSizeAttribute, None
                        )
                        if err_p == 0 and pos_val is not None:
                            ok_p, pt = AXValueGetValue(pos_val, kAXValueCGPointType, None)
                            ok_s, sz = (False, None)
                            if err_s == 0 and sz_val is not None:
                                ok_s, sz = AXValueGetValue(sz_val, kAXValueCGSizeType, None)

                            if ok_p:
                                elem_x = pt.x
                                elem_h = sz.height if (ok_s and sz.height > 0) else 24.0
                                elem_y = primary_h - pt.y - elem_h

                                target_x = elem_x + 8.0
                                target_y = elem_y - hud_h - 6.0
                                caret_y = elem_y
                                caret_h = elem_h
        except Exception as e:
            logger.debug(f"Could not locate caret via Accessibility: {e}")

    # Determine matching screen for the target coordinates
    if screens_list and target_x is not None and target_y is not None:
        test_pt = NSPoint(target_x + hud_w / 2.0, target_y + hud_h / 2.0)
        for s in screens_list:
            if NSPointInRect(test_pt, s.frame()):
                target_screen_frame = s.visibleFrame()
                break

    # If caret/focused element was located and target_y clips below that screen's bottom margin:
    if target_y is not None and target_y < target_screen_frame.origin.y + 10.0:
        if caret_h is not None and caret_y is not None:
            target_y = caret_y + caret_h + 6.0

    # 3. Fallback: Mouse cursor location
    if target_x is None or target_y is None:
        try:
            m_pt = NSEvent.mouseLocation()
            mouse_screen_frame = active_screen_frame
            if screens_list:
                for s in screens_list:
                    if NSPointInRect(m_pt, s.frame()):
                        mouse_screen_frame = s.visibleFrame()
                        break
            target_screen_frame = mouse_screen_frame
            target_x = m_pt.x - (hud_w / 2.0)
            target_y = m_pt.y - hud_h - 10.0
            if target_y < target_screen_frame.origin.y + 10.0:
                target_y = m_pt.y + 10.0
        except Exception as e:
            logger.debug(f"Cursor position fallback error: {e}")

    # 4. Ultimate fallback: Bottom-center of the screen
    if target_x is None or target_y is None:
        target_x = (
            target_screen_frame.origin.x
            + (target_screen_frame.size.width - hud_w) / 2.0
        )
        target_y = target_screen_frame.origin.y + 70.0

    # Ensure HUD remains fully inside the target screen visible bounds
    target_x = max(
        target_screen_frame.origin.x + 8.0,
        min(
            target_x,
            target_screen_frame.origin.x
            + target_screen_frame.size.width
            - hud_w
            - 8.0,
        ),
    )
    target_y = max(
        target_screen_frame.origin.y + 8.0,
        min(
            target_y,
            target_screen_frame.origin.y
            + target_screen_frame.size.height
            - hud_h
            - 8.0,
        ),
    )

    return NSPoint(target_x, target_y)


class VoiceWaveformView(NSView):
    """
    Compact, transparent dictaphone icon badge with 5 highly dynamic blue bars:
    - Pure transparent background without any surrounding container/pill.
    - Bars expand widely (up to 38px) when voice is active.
    - Completely still 4px resting capsules when silent.
    - Vibrant Electric Blue / Cyan gradient with drop shadow.
    """

    def initWithFrame_(self, frame):
        self = objc.super(VoiceWaveformView, self).initWithFrame_(frame)
        if self is None:
            return None

        self.num_bars = 5
        self.bar_heights = [4.0] * self.num_bars
        self.target_heights = [4.0] * self.num_bars
        self.level = 0.0
        self.smoothed_level = 0.0
        self.state = "idle"  # "recording", "transcribing", "idle"
        self.time_offset = 0.0

        # Dictaphone Electric Blue / Cyan Gradient
        self.color_blue = NSColor.colorWithRed_green_blue_alpha_(
            10 / 255, 132 / 255, 255 / 255, 1.0
        )
        self.color_cyan = NSColor.colorWithRed_green_blue_alpha_(
            80 / 255, 220 / 255, 255 / 255, 1.0
        )
        self.grad_blue = NSGradient.alloc().initWithStartingColor_endingColor_(
            self.color_blue, self.color_cyan
        )

        # Transcribing Electric Violet / Purple Gradient
        self.color_purple = NSColor.colorWithRed_green_blue_alpha_(
            180 / 255, 80 / 255, 240 / 255, 1.0
        )
        self.color_indigo = NSColor.colorWithRed_green_blue_alpha_(
            90 / 255, 90 / 255, 230 / 255, 1.0
        )
        self.grad_purple = NSGradient.alloc().initWithStartingColor_endingColor_(
            self.color_indigo, self.color_purple
        )

        # Soft drop shadow for high contrast over any wallpaper/window
        self.shadow = NSShadow.alloc().init()
        self.shadow.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.45))
        self.shadow.setShadowBlurRadius_(3.0)
        self.shadow.setShadowOffset_(NSSize(0.0, -1.0))

        return self

    def isOpaque(self) -> bool:
        return False

    def updateLevel_state_speed_(self, level: float, state: str, speed: float = 1.0):
        if not math.isfinite(level):
            level = 0.0
        if not math.isfinite(speed):
            speed = 1.0
        self.level = max(0.0, min(1.0, level))
        self.state = state
        self.speed = max(0.6, min(3.2, speed))

        # Snappy audio level tracking (reacts instantly to speech syllables)
        if self.level > self.smoothed_level:
            self.smoothed_level = self.smoothed_level * 0.15 + self.level * 0.85
        else:
            self.smoothed_level = self.smoothed_level * 0.55 + self.level * 0.45

        if self.smoothed_level < 0.010:
            self.smoothed_level = 0.0

        # Animation speed adapts dynamically to speech tempo & timbre
        if self.smoothed_level > 0.015:
            self.time_offset += 0.075 * self.speed
        elif self.state == "transcribing":
            self.time_offset += 0.080

        min_h = 4.5
        h = self.bounds().size.height if hasattr(self, "bounds") else 52.0
        max_h = min(44.0, max(12.0, h - 6.0)) if h > 12.0 else 32.0

        changed = False
        for i in range(self.num_bars):
            center_weight = CENTER_WEIGHTS[i]

            if self.state == "recording":
                if self.smoothed_level > 0.015:
                    # Dynamic wave tied directly to speech cadence & timbre
                    freq1 = math.sin(self.time_offset * 4.4 + i * 0.95)
                    freq2 = math.cos(self.time_offset * 7.0 - i * 0.70)
                    oscillation = 0.60 + 0.40 * (freq1 * 0.60 + freq2 * 0.40)
                    target = (
                        min_h
                        + (max_h - min_h)
                        * (self.smoothed_level**0.82)
                        * center_weight
                        * oscillation
                    )
                else:
                    # Silence ('молчок'): cleanly rests at minimum
                    target = min_h
            elif self.state == "transcribing":
                wave = (math.sin(self.time_offset * 5.0 + i * 0.90) + 1.0) * 0.5
                target = min_h + 18.0 * wave * center_weight
            else:
                target = min_h

            # Punchy, responsive spring interpolation (less sluggish, matches real dictaphone)
            new_h = self.bar_heights[i] * 0.22 + target * 0.78
            if abs(new_h - self.bar_heights[i]) > 0.05:
                changed = True
            self.bar_heights[i] = new_h

        if changed or self.state != "idle":
            self.setNeedsDisplay_(True)

    def updateLevel_state_(self, level: float, state: str):
        """Backwards compatibility forwarder."""
        self.updateLevel_state_speed_(level, state, 1.0)

    def drawRect_(self, dirtyRect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height

        is_transcribing = self.state == "transcribing"
        current_grad = self.grad_purple if is_transcribing else self.grad_blue

        NSGraphicsContext.saveGraphicsState()
        NSColor.clearColor().set()
        NSRectFillUsingOperation(dirtyRect, NSCompositingOperationClear)

        if w <= 40.0:
            # Miniature waveform indicator in bottom right corner of expanding Live HUD
            mini_bars = 4
            bar_w = 2.5
            spacing = 2.0
            total_w = mini_bars * bar_w + (mini_bars - 1) * spacing
            start_x = (w - total_w) / 2.0
            center_y = h / 2.0

            for i in range(mini_bars):
                bx = start_x + i * (bar_w + spacing)
                idx = min(i, len(self.bar_heights) - 1)
                norm_h = min(1.0, max(0.0, (self.bar_heights[idx] - 4.0) / 28.0))
                bh = max(3.0, min(h - 2.0, 3.0 + norm_h * (h - 5.0)))
                by = center_y - bh / 2.0

                bar_rect = NSRect(NSPoint(bx, by), NSSize(bar_w, bh))
                corner_r = min(bar_w / 2.0, bh / 2.0)
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    bar_rect, corner_r, corner_r
                )
                current_grad.drawInBezierPath_angle_(path, 90.0)
        else:
            # Full classic mode: 5 bold equalizer bars with soft drop shadow
            self.shadow.set()
            bar_w = 7.0
            spacing = 5.0
            total_w = self.num_bars * bar_w + (self.num_bars - 1) * spacing
            start_x = (w - total_w) / 2.0
            center_y = h / 2.0

            for i in range(self.num_bars):
                bx = start_x + i * (bar_w + spacing)
                bh = max(4.0, min(h - 2.0, self.bar_heights[i]))
                by = center_y - bh / 2.0

                bar_rect = NSRect(NSPoint(bx, by), NSSize(bar_w, bh))
                corner_r = min(bar_w / 2.0, bh / 2.0)
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    bar_rect, corner_r, corner_r
                )
                current_grad.drawInBezierPath_angle_(path, 90.0)

        NSGraphicsContext.restoreGraphicsState()


class LiveHUDGlassBackdropView(NSVisualEffectView):
    """
    Native macOS frosted glass view (NSVisualEffectView) with dynamic
    transparency / matte regulation (glass / balanced / matte) and crisp 1px border.
    Active strictly in Live Preview mode.
    """

    def initWithFrame_(self, frame):
        self = objc.super(LiveHUDGlassBackdropView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.glass_style = "glass"
        self.glass_opacity = 0.22

        self.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self.setState_(NSVisualEffectStateActive)
        self.setMaterial_(NSVisualEffectMaterialHUDWindow)
        self.setWantsLayer_(True)
        if self.layer() is not None:
            self.layer().setCornerRadius_(18.0)
            self.layer().setMasksToBounds_(True)
        return self

    def isOpaque(self) -> bool:
        return False

    def setGlassStyle_(self, style: str):
        self.glass_style = str(style)
        alphas = {
            "glass": 0.22,
            "balanced": 0.55,
            "matte": 0.85,
        }
        self.glass_opacity = alphas.get(self.glass_style, 0.22)
        self.setNeedsDisplay_(True)

    def drawRect_(self, dirtyRect):
        objc.super(LiveHUDGlassBackdropView, self).drawRect_(dirtyRect)

        NSGraphicsContext.saveGraphicsState()
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height

        bg_rect = NSRect(NSPoint(1.0, 1.0), NSSize(w - 2.0, h - 2.0))
        corner_r = 18.0
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bg_rect, corner_r, corner_r
        )

        # Tint fill according to transparency setting (frosted glass to opaque matte)
        NSColor.colorWithCalibratedWhite_alpha_(0.08, self.glass_opacity).setFill()
        path.fill()

        # Crisp subtle border
        border_alpha = 0.22 if self.glass_style == "glass" else 0.16
        NSColor.colorWithCalibratedWhite_alpha_(1.0, border_alpha).setStroke()
        path.setLineWidth_(1.0)
        path.stroke()
        NSGraphicsContext.restoreGraphicsState()


class LiveHUDContainerView(NSView):
    """
    Transparent container view for VoiceInputHUD.
    1. Live Preview mode: hosts LiveHUDGlassBackdropView for native frosted glass capsule.
    2. Classic mode: 100% transparent container with NO background, NO backdrop, NO square border,
       displaying only the dictaphone equalizer bars floating directly on screen.
    """

    def initWithFrame_(self, frame):
        self = objc.super(LiveHUDContainerView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.is_preview_enabled = True
        self.glass_style = "glass"
        self.glass_opacity = 0.22

        self.glass_backdrop = LiveHUDGlassBackdropView.alloc().initWithFrame_(frame)
        self.glass_backdrop.setGlassStyle_(self.glass_style)
        self.addSubview_(self.glass_backdrop)
        return self

    def isOpaque(self) -> bool:
        return False

    def setLivePreviewEnabled_(self, enabled: bool):
        self.is_preview_enabled = bool(enabled)
        if self.glass_backdrop is not None:
            self.glass_backdrop.setHidden_(not self.is_preview_enabled)
        self.setNeedsDisplay_(True)

    def setGlassStyle_(self, style: str):
        self.glass_style = str(style)
        alphas = {
            "glass": 0.22,
            "balanced": 0.55,
            "matte": 0.85,
        }
        self.glass_opacity = alphas.get(self.glass_style, 0.22)
        if self.glass_backdrop is not None:
            self.glass_backdrop.setGlassStyle_(self.glass_style)
        self.setNeedsDisplay_(True)

    def drawRect_(self, dirtyRect):
        # Pure transparent base container: never draws any background or border itself.
        # This guarantees 100% borderless/background-free rendering in classic mode.
        pass


class VoiceInputHUD(NSObject):
    """
    Floating voice animation HUD badge.
    Appears directly under the blinking text cursor.
    Supports:
    - Live Preview mode: Native macOS frosted glass capsule with dynamically expanding width/height,
      full phrase transcription, and a miniature voice activity equalizer in bottom-right corner.
    - Classic mode: 5 equalizer bars on a transparent background.
    """

    def init(self):
        self = objc.super(VoiceInputHUD, self).init()
        if self is None:
            return None

        self.panel = None
        self.container_view = None
        self.wave_view = None
        self.text_label = None
        self.timer = None
        self._fade_timer = None
        self.audio_recorder = None
        self.state = "idle"
        self._is_visible = False
        self.live_preview_enabled = True
        self.glass_style = "glass"
        self._current_text = "..."
        self.text_font = NSFont.systemFontOfSize_weight_(13.5, 0.2)

        self._create_panel()
        return self

    def _create_panel(self):
        width = PREVIEW_WIDTH if self.live_preview_enabled else CLASSIC_WIDTH
        height = PREVIEW_HEIGHT if self.live_preview_enabled else CLASSIC_HEIGHT

        rect = NSRect(NSPoint(100, 100), NSSize(width, height))
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, STYLE_MASK, NSBackingStoreBuffered, False
        )
        self.panel.setFloatingPanel_(True)
        self.panel.setLevel_(FLOATING_LEVEL)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(False)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setAlphaValue_(0.0)

        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
        )

        self.container_view = LiveHUDContainerView.alloc().initWithFrame_(
            NSRect(NSPoint(0.0, 0.0), NSSize(width, height))
        )
        self.container_view.setLivePreviewEnabled_(self.live_preview_enabled)
        self.container_view.setGlassStyle_(self.glass_style)

        # 1. Text preview label (left aligned with padding for bottom-right mini equalizer)
        pad_left = 14.0
        pad_right = 38.0
        avail_w = max(40.0, width - pad_left - pad_right)
        self.text_label = NSTextField.alloc().initWithFrame_(
            NSRect(NSPoint(pad_left, 10.0), NSSize(avail_w, height - 20.0))
        )
        self.text_label.setEditable_(False)
        self.text_label.setSelectable_(False)
        self.text_label.setBezeled_(False)
        self.text_label.setDrawsBackground_(False)
        self.text_label.setAlignment_(NSTextAlignmentLeft)
        self.text_label.setFont_(self.text_font)
        self.text_label.setTextColor_(NSColor.whiteColor())
        cell = self.text_label.cell()
        if cell is not None:
            cell.setWraps_(True)
            cell.setLineBreakMode_(NSLineBreakByWordWrapping)
            cell.setScrollable_(False)
        self.text_label.setHidden_(not self.live_preview_enabled)
        self.container_view.addSubview_(self.text_label)

        # 2. Waveform view (mini equalizer in bottom right when in preview mode; full size in classic mode)
        if self.live_preview_enabled:
            mini_w = 26.0
            mini_h = 14.0
            wave_frame = NSRect(
                NSPoint(width - mini_w - 8.0, 8.0), NSSize(mini_w, mini_h)
            )
        else:
            wave_frame = NSRect(NSPoint(0.0, 0.0), NSSize(width, height))

        self.wave_view = VoiceWaveformView.alloc().initWithFrame_(wave_frame)
        self.container_view.addSubview_(self.wave_view)

        self.panel.setContentView_(self.container_view)

    def _update_layout(self):
        if self.live_preview_enabled:
            display_text = self._current_text or "..."
            ns_str = NSString.stringWithString_(display_text)
            attrs = {NSFontAttributeName: self.text_font}
            single_sz = ns_str.sizeWithAttributes_(attrs)

            pad_left = 14.0
            pad_right = 38.0  # Room for bottom-right mini-waveform (26px + 12px margin)
            min_w = PREVIEW_MIN_WIDTH
            max_w = PREVIEW_MAX_WIDTH
            base_h = PREVIEW_HEIGHT

            if display_text == "..." or (
                single_sz.width + pad_left + pad_right <= PREVIEW_WIDTH
            ):
                w = PREVIEW_WIDTH
                h = base_h
                text_h = math.ceil(single_sz.height)
            elif single_sz.width + pad_left + pad_right <= max_w:
                w = math.ceil(single_sz.width + pad_left + pad_right)
                h = base_h
                text_h = math.ceil(single_sz.height)
            else:
                w = max_w
                avail_w = max_w - pad_left - pad_right
                options = (
                    NSStringDrawingUsesLineFragmentOrigin
                    | NSStringDrawingUsesFontLeading
                )
                rect = ns_str.boundingRectWithSize_options_attributes_context_(
                    NSSize(avail_w, 1200.0), options, attrs, None
                )
                text_h = math.ceil(rect.size.height)
                h = max(base_h, text_h + 20.0)

            if self.container_view is not None:
                self.container_view.setLivePreviewEnabled_(True)
                self.container_view.setFrame_(NSRect(NSPoint(0.0, 0.0), NSSize(w, h)))
                if hasattr(self.container_view, "glass_backdrop") and self.container_view.glass_backdrop is not None:
                    self.container_view.glass_backdrop.setFrame_(NSRect(NSPoint(0.0, 0.0), NSSize(w, h)))
                    self.container_view.glass_backdrop.setHidden_(False)
                self.container_view.setNeedsDisplay_(True)

            if self.text_label is not None:
                avail_w = w - pad_left - pad_right
                label_y = max(8.0, (h - text_h) / 2.0)
                self.text_label.setFrame_(
                    NSRect(NSPoint(pad_left, label_y), NSSize(avail_w, text_h + 4.0))
                )
                self.text_label.setHidden_(False)
                if self.state == "recording" and (
                    not self.text_label.stringValue()
                    or self.text_label.stringValue() == "Обработка..."
                ):
                    self.text_label.setStringValue_("...")
                    self.text_label.setTextColor_(
                        NSColor.colorWithCalibratedWhite_alpha_(0.7, 0.8)
                    )

            if self.wave_view is not None:
                mini_w = 26.0
                mini_h = 14.0
                self.wave_view.setFrame_(
                    NSRect(NSPoint(w - mini_w - 8.0, 8.0), NSSize(mini_w, mini_h))
                )
                self.wave_view.setHidden_(False)
                self.wave_view.setNeedsDisplay_(True)

        else:
            w = CLASSIC_WIDTH
            h = CLASSIC_HEIGHT

            if self.container_view is not None:
                self.container_view.setLivePreviewEnabled_(False)
                self.container_view.setFrame_(NSRect(NSPoint(0.0, 0.0), NSSize(w, h)))
                if hasattr(self.container_view, "glass_backdrop") and self.container_view.glass_backdrop is not None:
                    self.container_view.glass_backdrop.setHidden_(True)
                self.container_view.setNeedsDisplay_(True)

            if self.text_label is not None:
                self.text_label.setHidden_(True)
                self.text_label.setStringValue_("")

            if self.wave_view is not None:
                self.wave_view.setFrame_(NSRect(NSPoint(0.0, 0.0), NSSize(w, h)))
                self.wave_view.setHidden_(False)
                self.wave_view.setNeedsDisplay_(True)

        if self.panel is not None:
            if self._is_visible:
                new_pt = locate_caret_or_cursor(w, h)
                self.panel.setFrame_display_(NSRect(new_pt, NSSize(w, h)), True)
            else:
                cur_origin = self.panel.frame().origin
                self.panel.setFrame_display_(
                    NSRect(cur_origin, NSSize(w, h)), False
                )

    def setGlassStyle_(self, style: str):
        self.glass_style = str(style)
        if self.container_view is not None:
            self.container_view.setGlassStyle_(self.glass_style)

    def setLivePreviewEnabled_(self, enabled: bool):
        self.live_preview_enabled = bool(enabled)
        if not self.live_preview_enabled:
            self._current_text = ""
        self._update_layout()

    def setPreviewText_(self, text: str):
        if not self.live_preview_enabled or self.text_label is None:
            return
        if self.state != "recording":
            return
        clean = " ".join(text.split()) if text else ""
        self._current_text = clean if clean else "..."
        if clean:
            self.text_label.setStringValue_(clean)
            self.text_label.setTextColor_(NSColor.whiteColor())
        else:
            self.text_label.setStringValue_("...")
            self.text_label.setTextColor_(
                NSColor.colorWithCalibratedWhite_alpha_(0.7, 0.8)
            )
        self._update_layout()

    def setAudioRecorder_(self, recorder):
        self.audio_recorder = recorder

    def showRecording(self):
        """Locates the text cursor and displays HUD directly under it."""
        if self._fade_timer is not None:
            try:
                self._fade_timer.invalidate()
            except Exception as e:
                logger.debug(f"Fade timer invalidation error: {e}")
            self._fade_timer = None

        self.state = "recording"
        self._is_visible = True
        self._current_text = "..."
        if self.live_preview_enabled and self.text_label is not None:
            self.text_label.setStringValue_("...")
            self.text_label.setTextColor_(
                NSColor.colorWithCalibratedWhite_alpha_(0.7, 0.8)
            )
        self._update_layout()

        self.panel.orderFrontRegardless()
        self.panel.animator().setAlphaValue_(1.0)
        self._start_timer()

    def showTranscribing(self):
        """Switches HUD to transcribing state."""
        self.state = "transcribing"
        if self.live_preview_enabled and self.text_label is not None:
            cur_text = self.text_label.stringValue()
            if not cur_text or cur_text == "...":
                self._current_text = "Обработка..."
                self.text_label.setStringValue_("Обработка...")
                self.text_label.setTextColor_(
                    NSColor.colorWithCalibratedWhite_alpha_(0.85, 0.9)
                )
                self._update_layout()

    def hideHUD(self):
        """Smoothly fades out and closes HUD."""
        if not self._is_visible:
            return
        self._is_visible = False
        self.state = "idle"
        # Stop animation timer immediately to save CPU; AppKit animator handles window fade-out via GPU
        self._stop_timer()

        self.panel.animator().setAlphaValue_(0.0)
        if self._fade_timer is not None:
            try:
                self._fade_timer.invalidate()
            except Exception as e:
                logger.debug(f"Fade timer invalidation error: {e}")
            self._fade_timer = None

        self._fade_timer = (
            NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                0.18, self, "_onFadeDone:", None, False
            )
        )
        NSRunLoop.mainRunLoop().addTimer_forMode_(
            self._fade_timer, NSRunLoopCommonModes
        )

    def _onFadeDone_(self, timer):
        self._fade_timer = None
        if self._is_visible or self.state == "recording":
            return
        self._stop_timer()
        if self.panel:
            self.panel.orderOut_(None)
            if self.text_label is not None:
                self.text_label.setStringValue_("")

    def _start_timer(self):
        self._stop_timer()
        # 40 FPS (25ms) provides silky-smooth Retina rendering while cutting timer CPU wakeups by 38%
        self.timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            0.025, self, "onTimerTick:", None, True
        )
        NSRunLoop.mainRunLoop().addTimer_forMode_(self.timer, NSRunLoopCommonModes)

    def _stop_timer(self):
        if self.timer is not None:
            try:
                self.timer.invalidate()
            except Exception as e:
                logger.debug(f"HUD timer invalidation error: {e}")
            self.timer = None

    def onTimerTick_(self, timer):
        level = 0.0
        speed = 1.0
        if self.audio_recorder is not None and self.state == "recording":
            if hasattr(self.audio_recorder, "get_audio_state"):
                level, speed = self.audio_recorder.get_audio_state()
            else:
                level = self.audio_recorder.get_level()

        if self.wave_view:
            self.wave_view.updateLevel_state_speed_(level, self.state, speed)
