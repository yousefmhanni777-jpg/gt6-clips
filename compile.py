#!/usr/bin/env python3
"""Recopilaciones para YouTube: bugs, curiosidades o mejores momentos de streamers en GTA 6.

Coge los clips de esa categoría guardados en state.json, los ordena por vistas (el mejor al final),
monta un vídeo horizontal 1920x1080 con intro y rótulos, lo sube como Release de GitHub y
manda a Telegram el enlace de descarga con título, descripción y capítulos.

Uso:
  python compile.py                  # bugs, últimos 7 días, top 10
  python compile.py --cat curiosidad # también: todo (mejores momentos de cualquier categoría)
  python compile.py --days 3 --top 5
  python compile.py --dry-run        # deja el vídeo en out/ y no sube nada
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

import pipeline as p

CW, CH = 1920, 1080
INTRO_SECS = 3
KINDS = {
    "bug": {"title": "TOP {n} BUGS DE STREAMERS EN GTA 6", "about": "bugs más locos",
            "tags": ["#bugs", "#gta6bugs"], "emoji": "🐛"},
    "curiosidad": {"title": "{n} DETALLES DE GTA 6 QUE DESCUBRIERON LOS STREAMERS",
                   "about": "detalles, curiosidades y easter eggs más sorprendentes",
                   "tags": ["#curiosidades", "#eastereggs", "#gta6detalles"], "emoji": "🔍"},
    "todo": {"title": "LOS {n} MEJORES MOMENTOS DE STREAMERS EN GTA 6", "about": "mejores momentos",
             "tags": ["#mejoresmomentos", "#gta6momentos"], "emoji": "🔥"},
}
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
       "-video_track_timescale", "15360", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


def ass_doc(styles, events):
    head = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {CW}", f"PlayResY: {CH}", "WrapStyle: 0", "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"]
    body = ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    return "\n".join(head + styles + body + [
        f"Dialogue: 0,{p.ass_time(0)},{p.ass_time(600)},{st},,0,0,0,,{p.ass_escape(txt)}" for st, txt in events]) + "\n"


def box_style(name, font, size, color, align, margin_l, margin_v):
    c = p.ass_color(color)
    return (f"Style: {name},{font},{size},&H00000000,&H00000000,{c},{c},0,0,0,0,100,100,0,0,3,14,0,"
            f"{align},{margin_l},{margin_l},{margin_v},1")


def text_style(name, font, size, align, margin, alpha="00", margin_v=None):
    return (f"Style: {name},{font},{size},&H{alpha}FFFFFF,&H00FFFFFF,&H80000000,&H00000000,0,0,0,0,100,100,1,0,1,"
            f"4,0,{align},{margin},{margin},{margin if margin_v is None else margin_v},1")


def render_intro(work, title, subtitle, brand):
    ass = work / "intro.ass"
    ass.write_text(ass_doc(
        [box_style("T", brand["font"], 110, brand["color"], 5, 120, 0),
         text_style("S", brand["font"], 56, 2, 260)],
        [("T", title), ("S", subtitle)]), encoding="utf-8")
    out = work / "seg_000.mp4"
    p.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x111111:s={CW}x{CH}:r=30:d={INTRO_SECS}",
           "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={INTRO_SECS}",
           "-vf", f"ass={ass.name}:fontsdir={p.FONTS_DIR}", "-shortest", *ENC, str(out)], cwd=work)
    return out


def render_segment(work, idx, src, rank, clip, brand):
    ass = work / f"seg_{idx:03d}.ass"
    ass.write_text(ass_doc(
        [box_style("R", brand["font"], 64, brand["color"], 7, 50, 50),
         text_style("H", brand["font"], 54, 7, 50, margin_v=150),
         text_style("C", brand["font"], 38, 3, 40, alpha="30")],
        [("R", f"#{rank}  {clip['streamer'].upper()}"),
         ("H", clip["hook"].upper()),
         ("C", f"twitch.tv/{clip['login']}  ·  {brand['handle']}")]), encoding="utf-8")
    out = work / f"seg_{idx:03d}.mp4"
    vf = (f"scale={CW}:{CH}:force_original_aspect_ratio=decrease,pad={CW}:{CH}:(ow-iw)/2:(oh-ih)/2,"
          f"setsar=1,fps=30,ass={ass.name}:fontsdir={p.FONTS_DIR}")
    p.run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-af", "aresample=48000", *ENC, str(out)], cwd=work)
    return out


def duration(path):
    r = p.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(r.stdout.strip())


def mmss(t):
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def ai_texts(model, top, clips, about):
    lista = "\n".join(f"#{c['rank']} {c['streamer']}: {c['hook']}" for c in clips)
    prompt = (f"Escribe el título y la descripción de un vídeo de YouTube en español: recopilación de los TOP {top} "
              f"{about} de streamers jugando a GTA 6. Clips:\n{lista}\n\n"
              "Título: máx 70 caracteres, con gancho, menciona a 1-2 streamers famosos de la lista. "
              "Descripción: 2-3 frases naturales, invita a suscribirse. "
              'Responde SOLO con JSON: {"titulo": "...", "descripcion": "..."}')
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {p.env('GROQ_API_KEY')}"},
                      json={"model": model, "temperature": 0.7, "response_format": {"type": "json_object"},
                            "messages": [{"role": "user", "content": prompt}]}, timeout=60)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def publish_release(video, name, body):
    repo, token = p.env("GITHUB_REPOSITORY"), p.env("GITHUB_TOKEN")
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    tag = f"recopilacion-{datetime.now():%Y%m%d-%H%M}"
    r = requests.post(f"https://api.github.com/repos/{repo}/releases", headers=h,
                      json={"tag_name": tag, "name": name, "body": body}, timeout=60)
    r.raise_for_status()
    rel = r.json()
    with open(video, "rb") as f:
        r = requests.post(f"https://uploads.github.com/repos/{repo}/releases/{rel['id']}/assets",
                          params={"name": video.name}, data=f, timeout=1800,
                          headers={**h, "Content-Type": "video/mp4"})
    r.raise_for_status()
    return r.json()["browser_download_url"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cat", choices=list(KINDS), default="bug")
    ap.add_argument("--min", type=int, default=3, help="mínimo de clips para montar el vídeo")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((p.ROOT / "config.yaml").read_text())
    brand = cfg["brand"]
    state = p.load_state()
    cutoff = time.time() - args.days * 86400
    kind = KINDS[args.cat]
    clips = [{"id": k, **v} for k, v in state["processed"].items()
            if v.get("t", 0) > cutoff and v.get("duration", 0) >= 10 and v.get("url")
            and (v.get("cat") == args.cat if args.cat != "todo" else v.get("status") == "enviado")]
    clips = sorted(clips, key=lambda c: c.get("views", 0), reverse=True)[:args.top]
    if len(clips) < args.min:
        msg = (f"{kind['emoji']} Recopilación '{args.cat}': solo hay {len(clips)} clips en los últimos "
               f"{args.days:g} días (mínimo {args.min}). No se monta vídeo todavía.")
        p.log(msg)
        if not args.dry_run:
            p.send_telegram_text(msg)
        return

    # Cuenta atrás: el más visto, al final.
    clips.reverse()
    for i, c in enumerate(clips):
        c["rank"] = len(clips) - i

    work = p.WORK_DIR / "compilacion"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        segs = [render_intro(work, kind["title"].format(n=len(clips)),
                             f"Semana del {datetime.now():%d/%m/%Y}  ·  {brand['handle']}", brand)]
        chapters, t = ["0:00 Intro"], float(INTRO_SECS)
        ok = []
        for i, c in enumerate(clips, start=1):
            try:
                src = work / f"src_{i:03d}.mp4"
                p.download(c["url"], src)
                seg = render_segment(work, i, src, c["rank"], c, brand)
            except Exception as e:
                p.log(f"[{c['streamer']}] se salta: {e}")
                continue
            chapters.append(f"{mmss(t)} #{c['rank']} {c['streamer']} - {c['hook']}")
            t += duration(seg)
            segs.append(seg)
            ok.append(c)
        if len(ok) < args.min:
            raise RuntimeError(f"solo se pudieron montar {len(ok)} clips")

        (work / "list.txt").write_text("".join(f"file '{s.name}'\n" for s in segs))
        p.OUT_DIR.mkdir(exist_ok=True)
        out = p.OUT_DIR / f"top_{args.cat}_{datetime.now():%Y%m%d}.mp4"
        p.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
               "-movflags", "+faststart", str(out)], cwd=work)
        p.log(f"recopilación: {out.name} · {mmss(duration(out))} · {out.stat().st_size // 2**20} MB")

        texts = ai_texts(cfg["groq"]["llm_model"], len(ok), ok, kind["about"])
        credits = "\n".join(sorted({f"twitch.tv/{c['login']}" for c in ok}))
        description = (f"{texts.get('descripcion', '')}\n\n⏱️ Capítulos\n" + "\n".join(chapters) +
                       f"\n\n🎥 Créditos a los streamers:\n{credits}\n\n"
                       + " ".join(brand["hashtags"] + kind["tags"]))
        title = texts.get("titulo", kind["title"].format(n=len(ok)))
        if args.dry_run:
            p.log(f"--- {title} ---\n{description}")
            return
        url = publish_release(out, title, description)
        p.send_telegram_text(f"🎬 {kind['emoji']} RECOPILACIÓN LISTA ({mmss(duration(out))})\n\n⬇️ Descargar vídeo:\n{url}\n\n"
                             f"📋 TÍTULO:\n{title}\n\n📋 DESCRIPCIÓN:\n{description}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
