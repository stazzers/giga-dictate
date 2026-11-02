import logging
import math

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSEvent,
    NSGradient,
    NSGraphicsContext,
    NSPanel,
    NSPoint,
    NSRect,
    NSScreen,
    NSShadow,
    NSSize,
    NSView,
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

NUM_BARS = 5
CENTER_WEIGHTS = tuple(
    math.sin(((i + 0.5) / float(NUM_BARS)) * math.pi) ** 0.82 for i in range(NUM_BARS)
)


def locate_caret_or_cursor(hud_w: float = 72.0, hud_h: float = 52.0) -> NSPoint:
    """
    Finds screen coordinates (in Cocoa coordinate space) for:
    1. The blinking text cursor (caret) of the active input field (Safari, Notes, Word, Telegram, etc.)
    2. Fallback: Focused element bounds
    3. Fallback: Current mouse cursor position
    4. Fallback: Bottom-center of the screen
    """
    screen = NSScreen.mainScreen()
    screens = NSScreen.screens()
    primary_h = screens[0].frame().size.height if screens else 900.0
    screen_frame = (
        screen.visibleFrame() if screen else NSRect(NSPoint(0, 0), NSSize(1440, 900))
    )

    target_x = None
    target_y = None

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

                            # If clipping below bottom screen margin, place directly above the caret
                            if target_y < screen_frame.origin.y + 10.0:
                                target_y = caret_y + caret_h + 6.0

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
                            if target_y < screen_frame.origin.y + 10.0:
                                target_y = elem_y + elem_h + 6.0
    except Exception as e:
        logger.debug(f"Could not locate caret via Accessibility: {e}")

    # 3. Fallback: Mouse cursor location
    if target_x is None or target_y is None:
        try:
            m_pt = NSEvent.mouseLocation()
            if m_pt.x >= screen_frame.origin.x and m_pt.x <= (
                screen_frame.origin.x + screen_frame.size.width
            ):
                target_x = m_pt.x - (hud_w / 2.0)
                target_y = m_pt.y - hud_h - 10.0
                if target_y < screen_frame.origin.y + 10.0:
                    target_y = m_pt.y + 10.0
        except Exception as e:
            logger.debug(f"Cursor position fallback error: {e}")

    # 4. Ultimate fallback: Bottom-center of the screen
    if target_x is None or target_y is None:
        target_x = screen_frame.origin.x + (screen_frame.size.width - hud_w) / 2.0
        target_y = screen_frame.origin.y + 70.0

    # Ensure HUD remains fully inside the screen visible bounds
    target_x = max(
        screen_frame.origin.x + 8.0,
        min(target_x, screen_frame.origin.x + screen_frame.size.width - hud_w - 8.0),
    )
    target_y = max(
        screen_frame.origin.y + 8.0,
        min(target_y, screen_frame.origin.y + screen_frame.size.height - hud_h - 8.0),
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
        self.level = level
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
        max_h = 44.0  # Slightly larger opening as requested

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

        bar_w = 7.0  # Wider, bolder bars
        spacing = 5.0
        total_w = self.num_bars * bar_w + (self.num_bars - 1) * spacing
        start_x = (w - total_w) / 2.0
        center_y = h / 2.0

        is_transcribing = self.state == "transcribing"
        current_grad = self.grad_purple if is_transcribing else self.grad_blue

        NSGraphicsContext.saveGraphicsState()
        self.shadow.set()

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


class VoiceInputHUD(NSObject):
    """
    Compact, borderless floating voice animation badge.
    Appears directly under the blinking text cursor on a transparent background.
    """

    def init(self):
        self = objc.super(VoiceInputHUD, self).init()
        if self is None:
            return None

        self.panel = None
        self.wave_view = None
        self.timer = None
        self._fade_timer = None
        self.audio_recorder = None
        self.state = "idle"
        self._is_visible = False

        self._create_panel()
        return self

    def _create_panel(self):
        width = 72.0
        height = 52.0

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

        self.wave_view = VoiceWaveformView.alloc().initWithFrame_(
            NSRect(NSPoint(0.0, 0.0), NSSize(width, height))
        )
        self.panel.setContentView_(self.wave_view)

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

        w = self.panel.frame().size.width
        h = self.panel.frame().size.height
        caret_pt = locate_caret_or_cursor(w, h)
        self.panel.setFrameOrigin_(caret_pt)

        self.panel.orderFrontRegardless()
        self.panel.animator().setAlphaValue_(1.0)
        self._start_timer()

    def showTranscribing(self):
        """Switches HUD to transcribing state."""
        self.state = "transcribing"

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
        self._stop_timer()
        if not self._is_visible and self.panel:
            self.panel.orderOut_(None)

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
