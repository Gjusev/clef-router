"""Generate every visual asset for clef-router.

Outputs (committed):
    assets/logo.png            512x512 mark
    assets/social-preview-art.png  AI-generated routing illustration source
    assets/social-preview.png  1280x640 repository social card
    docs/how-it-works.svg      static routing diagram for the README
    docs/how-it-works.html     self-contained animated explainer page
    brag-output/brag.mp4       12s product demo (1280x720, h264)
    brag-output/brag.jpg       poster frame

Style: warm monochrome, hairline borders, muted pastels, no gradients,
no shadows, no emoji. Requires Pillow and ffmpeg on PATH.

Usage: python scripts/generate_assets.py [--skip-video]
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# Warm monochrome palette + muted pastels.
BONE = "#F7F6F3"
CARD = "#FFFFFF"
INK = "#111111"
MUTED = "#787774"
HAIRLINE = "#E6E4E0"
GREEN_BG, GREEN_FG = "#EDF3EC", "#346538"
RED_BG, RED_FG = "#FDEBEC", "#9F2F2D"
BLUE_BG, BLUE_FG = "#E1F3FE", "#1F6C9F"
YELLOW_BG, YELLOW_FG = "#FBF3DB", "#956400"

SERIF, SERIF_BOLD = "georgia.ttf", "georgiab.ttf"
MONO, MONO_BOLD = "consola.ttf", "consolab.ttf"
SANS, SANS_SEMI = "segoeui.ttf", "seguisb.ttf"

EASE = 0.16  # ease-out cubic exponent helper


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(name, size)


def ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (r, g, b, int(round(255 * min(1.0, max(0.0, alpha)))))


def rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int = 12,
    fill: str | None = CARD,
    outline: str = HAIRLINE,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def paste_layer(base: Image.Image, layer: Image.Image) -> None:
    base.alpha_composite(layer)


# --------------------------------------------------------------------------
# Logo
# --------------------------------------------------------------------------


def make_logo(path: Path) -> None:
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = 56
    d.rounded_rectangle(
        (m, m, size - m, size - m), radius=96, fill=BONE, outline=INK, width=8
    )
    cx = 150
    d.ellipse((cx - 26, 256 - 26, cx + 26, 256 + 26), fill=INK)
    d.line((cx + 26, 256, 268, 256), fill=INK, width=8)
    # gate: small square outline at the fork
    d.rounded_rectangle((268, 216, 340, 296), radius=18, outline=INK, width=8)
    # branches
    d.line((340, 236, 392, 170), fill=INK, width=8)
    d.line((340, 276, 392, 342), fill=INK, width=8)
    d.rounded_rectangle((388, 128, 448, 188), radius=16, fill=GREEN_BG, outline=GREEN_FG, width=6)
    d.rounded_rectangle((388, 324, 448, 384), radius=16, fill=RED_BG, outline=RED_FG, width=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# --------------------------------------------------------------------------
# Social preview
# --------------------------------------------------------------------------


def tier_pill(d: ImageDraw.ImageDraw, box, label, sub, bg, fg) -> None:
    rounded_card(d, box, radius=14, fill=bg, outline=fg, width=2)
    x0, y0 = box[0] + 18, box[1] + 10
    d.text((x0, y0), label, font=font(MONO_BOLD, 24), fill=fg)
    d.text((x0, y0 + 30), sub, font=font(MONO, 17), fill=MUTED)


def make_social_preview(path: Path) -> None:
    W, H = 1280, 640
    art = Image.open(ROOT / "assets" / "social-preview-art.png").convert("RGB")
    img = art.resize((W, H), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(img)
    # The generated visual leaves the left third clear. This panel keeps the
    # repository name readable in small social-feed previews.
    d.rectangle((14, 14, 620, H - 14), fill=BONE)
    d.rectangle((0, 0, W - 1, H - 1), outline=INK, width=2)
    d.rectangle((14, 14, 620, H - 14), outline=HAIRLINE, width=2)

    d.text((62, 78), "CLOUDFLARE CLEF  /  DECISION MODEL", font=font(MONO, 19), fill=MUTED)
    d.text((56, 118), "clef-router", font=font(SERIF_BOLD, 88), fill=INK)
    d.line((62, 238, 552, 238), fill=HAIRLINE, width=2)
    d.text(
        (62, 264),
        "Route each prompt to the affordable model\nwhen it is enough, and to the frontier\nmodel when it matters.",
        font=font(SANS, 27), fill="#2F3437", spacing=9,
    )
    d.text(
        (72, 470),
        "OpenAI-compatible proxy  ·  $0.03 per 1k decisions  ·  Apache-2.0",
        font=font(MONO, 22), fill=MUTED,
    )
    d.text(
        (72, 540),
        "BFCL 98.8 (clef-flash) vs 38.1 (laya)  —  Decision Index 0.2.1",
        font=font(MONO, 22), fill=BLUE_FG,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# --------------------------------------------------------------------------
# Static SVG diagram
# --------------------------------------------------------------------------

HOW_IT_WORKS_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 420"
     font-family="Consolas, 'SF Mono', monospace" role="img"
     aria-label="clef-router routes a prompt through the Clef decision model to the cheap or frontier tier">
  <rect width="960" height="420" fill="#F7F6F3"/>
  <rect x="1" y="1" width="958" height="418" fill="none" stroke="#E6E4E0" stroke-width="2"/>

  <g>
    <rect x="48" y="160" width="220" height="96" rx="14" fill="#FFFFFF" stroke="#E6E4E0" stroke-width="2"/>
    <text x="72" y="196" font-size="19" fill="#787774">client prompt</text>
    <text x="72" y="226" font-size="20" fill="#111111">"Refund order #4512"</text>
  </g>

  <line x1="268" y1="208" x2="330" y2="208" stroke="#111111" stroke-width="3"/>
  <polygon points="330,202 330,214 342,208" fill="#111111"/>

  <g>
    <rect x="346" y="158" width="200" height="100" rx="14" fill="#111111"/>
    <text x="398" y="200" font-size="22" font-weight="bold" fill="#F7F6F3">CLEF</text>
    <text x="374" y="230" font-size="15" fill="#B9B7B2">one forward pass</text>
  </g>

  <path d="M 546 192 L 610 148" stroke="#346538" stroke-width="3" fill="none"/>
  <path d="M 546 224 L 610 268" stroke="#9F2F2D" stroke-width="3" fill="none"/>

  <g>
    <rect x="614" y="96" width="300" height="96" rx="14" fill="#EDF3EC" stroke="#346538" stroke-width="2"/>
    <text x="638" y="132" font-size="20" font-weight="bold" fill="#346538">CHEAP tier</text>
    <text x="638" y="162" font-size="16" fill="#787774">greetings, lookups, rewrites</text>
  </g>
  <g>
    <rect x="614" y="222" width="300" height="96" rx="14" fill="#FDEBEC" stroke="#9F2F2D" stroke-width="2"/>
    <text x="638" y="258" font-size="20" font-weight="bold" fill="#9F2F2D">FRONTIER tier</text>
    <text x="638" y="288" font-size="16" fill="#787774">reasoning, code, incidents</text>
  </g>

  <text x="48" y="368" font-size="16" fill="#787774">gate: confidence below 0.45 escalates; unknown answers escalate; urgent prompts escalate</text>
</svg>
"""


def make_svg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HOW_IT_WORKS_SVG, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# Animated HTML explainer
# --------------------------------------------------------------------------

HOW_IT_WORKS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How clef-router works</title>
<style>
  :root {
    --bone:#F7F6F3; --card:#FFFFFF; --ink:#111111; --muted:#787774;
    --line:#E6E4E0; --green-bg:#EDF3EC; --green-fg:#346538;
    --red-bg:#FDEBEC; --red-fg:#9F2F2D; --blue-bg:#E1F3FE; --blue-fg:#1F6C9F;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--bone); color: var(--ink);
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    display: grid; place-items: center; min-height: 100dvh; padding: 48px 24px;
  }
  main { max-width: 920px; width: 100%; }
  .overline { font-family: Consolas, monospace; font-size: 13px; letter-spacing: .08em;
              color: var(--muted); text-transform: uppercase; }
  h1 { font-family: Georgia, serif; letter-spacing: -0.03em; font-size: 44px;
       line-height: 1.08; margin: 10px 0 6px; }
  .sub { color: var(--muted); max-width: 60ch; line-height: 1.6; }
  .stage {
    margin-top: 36px; background: var(--card); border: 1px solid var(--line);
    border-radius: 16px; padding: 32px; display: grid;
    grid-template-columns: 1fr auto 1fr; gap: 24px; align-items: center;
  }
  .prompt {
    border: 1px solid var(--line); border-radius: 12px; background: var(--bone);
    font-family: Consolas, monospace; font-size: 14px; padding: 14px 16px; min-height: 84px;
  }
  .prompt b { display: block; font-size: 11px; color: var(--muted);
              letter-spacing: .08em; margin-bottom: 8px; font-weight: 600; }
  .gate {
    border-radius: 12px; background: var(--ink); color: var(--bone);
    font-family: Consolas, monospace; font-weight: bold; padding: 18px 22px;
    text-align: center; position: relative;
  }
  .gate small { display: block; font-weight: normal; color: #B9B7B2; font-size: 11px; margin-top: 6px; }
  .dot {
    position: absolute; top: -10px; left: 50%; width: 10px; height: 10px;
    margin-left: -5px; border-radius: 50%; background: var(--blue-fg);
    animation: drop 2.8s infinite;
  }
  @keyframes drop {
    0%, 12% { transform: translate(-90px, 26px); opacity: 0; }
    22% { transform: translate(-90px, 26px); opacity: 1; }
    42% { transform: translate(0, 26px); opacity: 1; }
    60% { transform: translate(0, 40px); opacity: 1; }
    78%, 100% { transform: translate(0, 40px); opacity: 0; }
  }
  .tiers { display: grid; gap: 14px; }
  .tier {
    border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
    font-family: Consolas, monospace; font-size: 13px; transition: opacity .3s;
  }
  .tier b { display: block; font-size: 14px; letter-spacing: .06em; margin-bottom: 4px; }
  .cheap { background: var(--green-bg); border-color: var(--green-fg); color: var(--green-fg); }
  .frontier { background: var(--red-bg); border-color: var(--red-fg); color: var(--red-fg); }
  .tier .why { color: var(--muted); }
  .trace {
    grid-column: 1 / -1; font-family: Consolas, monospace; font-size: 12px;
    color: var(--muted); border-top: 1px solid var(--line); padding-top: 16px;
  }
  .trace em { font-style: normal; color: var(--ink); }
  .demo { animation: switch 8.4s infinite; }
  .demo.s2 { animation-delay: 2.8s; }
  .demo.s3 { animation-delay: 5.6s; }
  @keyframes switch {
    0%, 100% { opacity: 0; transform: translateY(8px); }
    6%, 88% { opacity: 1; transform: translateY(0); }
  }
  .scene { position: relative; }
  .scene > * { position: absolute; inset: 0; }
  .scene { min-height: 110px; }
  .card { height: 100%; }
  @media (prefers-reduced-motion: reduce) {
    .dot, .demo { animation: none; }
    .demo.s2, .demo.s3 { display: none; }
  }
</style>
</head>
<body>
<main>
  <p class="overline">How it works</p>
  <h1>One call to Clef,<br>every prompt routed</h1>
  <p class="sub">Your OpenAI client posts to <code>/v1/chat/completions</code>. clef-router
  turns the last user message into a Clef state, asks one joint question set
  (team + urgency), and answers with the tier decision. The completion model
  is yours to call, exactly like laya-router.</p>

  <section class="stage">
    <div class="scene">
      <div class="demo s1">
        <div class="prompt card"><b>PROMPT 1 / 3</b>"Hi! What's the capital of France?"</div>
      </div>
      <div class="demo s2">
        <div class="prompt card"><b>PROMPT 2 / 3</b>"Implement a token bucket rate limiter in Go."</div>
      </div>
      <div class="demo s3">
        <div class="prompt card"><b>PROMPT 3 / 3</b>"Checkout is down for every customer right now."</div>
      </div>
    </div>
    <div class="gate"><span class="dot"></span>CLEF<small>team + urgency,<br>one pass</small></div>
    <div class="tiers">
      <div class="tier cheap"><b>CHEAP</b><span class="why">routine, low stakes</span></div>
      <div class="tier frontier"><b>FRONTIER</b><span class="why">code, incidents, reasoning</span></div>
    </div>
    <p class="trace">decisions:
    <em>1) cheap, confidence 0.93</em> &nbsp; <em>2) frontier, code</em> &nbsp;
    <em>3) frontier, urgent (noul 0.97)</em> &nbsp; gate: confidence &lt; 0.45 escalates</p>
  </section>
</main>
</body>
</html>
"""


def make_html(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HOW_IT_WORKS_HTML, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# Brag video
# --------------------------------------------------------------------------

W, H = 1280, 720
SCALE = 2
FPS = 30

SCENE_END = {"title": 2.4, "demo": 9.6, "stats": 12.0}

DEMO_PROMPTS = [
    ("\"Hi! What's the capital of France?\"", "cheap", "confidence 0.93"),
    ("\"Implement a token bucket limiter in Go.\"", "frontier", "confidence 0.88"),
    ("\"Checkout is down for every customer.\"", "frontier", "urgent noul 0.97"),
    ("\"Translate 'thank you' to Japanese.\"", "cheap", "confidence 0.90"),
]


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W * SCALE, H * SCALE), BONE)
    return img, ImageDraw.Draw(img)


def fade_alpha(t: float, start: float, dur: float = 0.5) -> float:
    return ease_out((t - start) / dur)


def draw_frame(index: int) -> Image.Image:
    t = index / FPS
    base, d = new_frame()
    s = SCALE

    if t < SCENE_END["title"]:
        a = fade_alpha(t, 0.1)
        over = font(MONO, int(20 * s))
        title = font(SERIF_BOLD, int(96 * s))
        tag = font(SANS, int(28 * s))
        meta = font(MONO, int(20 * s))
        y = int(lerp(280, 260, ease_out(t / 1.0))) * s // s
        d.text((120 * s, y), "CLOUDFLARE CLEF / OPENAI-COMPATIBLE ROUTING", font=over, fill=rgba(MUTED, a))
        d.text((112 * s, y + 44 * s), "clef-router", font=title, fill=rgba(INK, a))
        d.line((120 * s, (y + 180) * s, 520 * s, (y + 180) * s), fill=rgba(HAIRLINE, a), width=2 * s)
        d.text(
            (120 * s, (y + 210) * s),
            "The affordable model when it is enough.\nThe frontier model when it matters.",
            font=tag, fill=rgba("#2F3437", a), spacing=8 * s,
        )
        d.text(
            (120 * s, (y + 330) * s),
            "policy accuracy 100% on committed fixtures (n=42)",
            font=meta, fill=rgba(BLUE_FG, a),
        )
    elif t < SCENE_END["demo"]:
        a = fade_alpha(t, SCENE_END["title"] - 0.4)
        local = t - SCENE_END["title"]
        slot = int(local // 1.8) % len(DEMO_PROMPTS)
        phase = (local % 1.8) / 1.8
        prompt, tier, note = DEMO_PROMPTS[slot]

        d.text((120 * s, 84 * s), "LIVE ROUTING", font=font(MONO, int(20 * s)), fill=rgba(MUTED, a))

        # prompt card slides in
        slide = int((1.0 - ease_out(min(1.0, phase * 2.2))) * 40 * s)
        rounded_card(d, (120 * s - slide, 150 * s, 620 * s - slide, 260 * s), radius=16 * s,
                     fill=rgba(CARD, a), outline=rgba(HAIRLINE, a), width=2 * s)
        d.text(((148) * s - slide, 176 * s), "PROMPT", font=font(MONO_BOLD, int(16 * s)),
               fill=rgba(MUTED, a))
        d.text(((148) * s - slide, 206 * s), prompt, font=font(MONO, int(22 * s)),
               fill=rgba(INK, a))

        # gate
        pulse = 1.0 + 0.03 * math.sin(phase * math.pi * 2)
        gw, gh = 200 * int(pulse * s) // s, 96 * s
        gx = (W // 2 - gw // 2) * s
        rounded_card(d, (gx, 320 * s, gx + gw * s // s * s + 0, 320 * s + gh), radius=14 * s,
                     fill=rgba(INK, a), outline=rgba(INK, a), width=2 * s)
        d.text((W // 2 * s - 44 * s, 340 * s), "CLEF", font=font(MONO_BOLD, int(26 * s)),
               fill=rgba(BONE, a))
        d.text((W // 2 * s - 88 * s, 380 * s), "team + urgency", font=font(MONO, int(15 * s)),
               fill=rgba("#B9B7B2", a))

        chosen_green = tier == "cheap"
        # connector line grows during first half of the slot
        grow = ease_out(min(1.0, phase * 1.8))
        color = GREEN_FG if chosen_green else RED_FG
        mid_y = 430 * s
        x0, x1 = W // 2 * s, (880 if chosen_green else 880) * s
        # vertical from gate, then horizontal to chosen tier
        v_end = int(lerp(416 * s, mid_y, grow))
        d.line((W // 2 * s, 416 * s, W // 2 * s, v_end), fill=rgba(color, a), width=4 * s)
        if grow > 0.99:
            d.line((min(x0, x1), mid_y, max(x0, x1), mid_y), fill=rgba(color, a), width=4 * s)
            d.polygon(
                [(x1 - 10 * s, mid_y - 7 * s), (x1 - 10 * s, mid_y + 7 * s), (x1 + 4 * s, mid_y)],
                fill=rgba(color, a),
            )

        # tier chips
        chip_alpha = a if grow > 0.6 else a * 0.5
        rounded_card(d, (700 * s, 470 * s, 1060 * s, 560 * s), radius=16 * s,
                     fill=rgba(GREEN_BG if chosen_green else CARD, chip_alpha),
                     outline=rgba(GREEN_FG if chosen_green else HAIRLINE, chip_alpha), width=2 * s)
        rounded_card(d, (700 * s, 580 * s, 1060 * s, 670 * s), radius=16 * s,
                     fill=rgba(RED_BG if not chosen_green else CARD, chip_alpha),
                     outline=rgba(RED_FG if not chosen_green else HAIRLINE, chip_alpha), width=2 * s)
        d.text((724 * s, 494 * s), "CHEAP", font=font(MONO_BOLD, int(24 * s)),
               fill=rgba(GREEN_FG, chip_alpha))
        d.text((724 * s, 528 * s), "8B class", font=font(MONO, int(17 * s)),
               fill=rgba(MUTED, chip_alpha))
        d.text((724 * s, 604 * s), "FRONTIER", font=font(MONO_BOLD, int(24 * s)),
               fill=rgba(RED_FG, chip_alpha))
        d.text((724 * s, 638 * s), "reasoning, code", font=font(MONO, int(17 * s)),
               fill=rgba(MUTED, chip_alpha))

        if phase > 0.55:
            note_a = fade_alpha(phase, 0.55, 0.25)
            d.text((120 * s, 620 * s),
                   f"decision: {tier:<8} {note}",
                   font=font(MONO_BOLD, int(22 * s)),
                   fill=rgba(INK if chosen_green else RED_FG, note_a))
    else:
        a = fade_alpha(t, SCENE_END["demo"] - 0.2)
        d.text((120 * s, 100 * s), "MEASURED, NOT PROMISED", font=font(MONO, int(20 * s)),
               fill=rgba(MUTED, a))
        stats = [
            ("100%", "policy accuracy", "committed fixtures, n=42"),
            ("0.12 ms", "decision overhead", "library p50, excludes network"),
            ("$0.033", "per 1k decisions", "input tokens at $0.24 / Mtok"),
        ]
        x = 120 * s
        for value, label, detail in stats:
            rounded_card(d, (x, 180 * s, x + 320 * s, 420 * s), radius=18 * s,
                         fill=rgba(CARD, a), outline=rgba(HAIRLINE, a), width=2 * s)
            d.text((x + 28 * s, 216 * s), value, font=font(SERIF_BOLD, int(64 * s)),
                   fill=rgba(INK, a))
            d.text((x + 28 * s, 320 * s), label, font=font(SANS_SEMI, int(22 * s)),
                   fill=rgba("#2F3437", a))
            d.text((x + 28 * s, 356 * s), detail, font=font(MONO, int(15 * s)),
                   fill=rgba(MUTED, a))
            x += 352 * s
        d.text((120 * s, 470 * s),
               "pip install clef-router   ·   clef-router --port 8000",
               font=font(MONO, int(22 * s)), fill=rgba(INK, a))
        d.text((120 * s, 510 * s),
               "github.com/Gjusev/clef-router   ·   Apache-2.0",
               font=font(MONO, int(20 * s)), fill=rgba(MUTED, a))

    # downscale for anti-aliasing
    return base.resize((W, H), Image.LANCZOS)


def make_video(out_dir: Path, skip: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if not skip:
        frames_dir.mkdir(parents=True, exist_ok=True)
        total = int(SCENE_END["stats"] * FPS)
        for i in range(total):
            draw_frame(i).save(frames_dir / f"f{i:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(FPS),
                "-i", str(frames_dir / "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(out_dir / "brag.mp4"),
            ],
            check=True, capture_output=True,
        )
        shutil.copy(frames_dir / f"f{int(7.2 * FPS):04d}.png", out_dir / "brag.jpg")
        shutil.rmtree(frames_dir)
    share = out_dir / "share-copy.txt"
    share.write_text(
        "clef-router: route each prompt to the affordable model when it is "
        "enough, and to the frontier model when it matters. OpenAI-compatible, "
        "powered by Cloudflare's Clef decision model.\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    make_logo(ROOT / "assets" / "logo.png")
    make_social_preview(ROOT / "assets" / "social-preview.png")
    make_svg(ROOT / "docs" / "how-it-works.svg")
    make_html(ROOT / "docs" / "how-it-works.html")
    make_video(ROOT / "brag-output", args.skip_video)
    print("assets generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
