#!/usr/bin/env python3
"""Formato "cara arriba, juego abajo": detecta la cámara del streamer y monta el vídeo 9:16 en dos pisos.

Uso (prueba, no toca el bot):
  python facecam.py --n 8            # rehace con este formato los últimos clips de reacción/gracioso enviados
  python facecam.py --clip URL       # un clip concreto
Los vídeos llegan a Telegram marcados como PRUEBA.
"""
import argparse
import json
import statistics
import time

import pipeline as p

TOP_H = 760                 # piso de arriba: cámara del streamer
BOT_H = p.H - TOP_H         # piso de abajo: juego
SAMPLES = 12                # fotogramas que se miran para encontrar la cara


def detect_cam(video):
    """Recuadro (x, y, w, h) de la cámara del streamer en el vídeo original, o None si no hay una cara fija."""
    import cv2  # solo hace falta en este formato

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []  # una lista de caras por fotograma
    for i in range(SAMPLES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 0.5) / SAMPLES))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(fh // 16, fh // 16))
        faces.append([tuple(int(v) for v in f) for f in found])
    cap.release()
    if not faces:
        return None
    # La cara de la cámara está casi quieta en todos los fotogramas; las del juego se mueven o aparecen y desaparecen.
    best, best_hits = None, 0
    for frame_faces in faces:
        for (x, y, w, h) in frame_faces:
            cx, cy = x + w / 2, y + h / 2
            hits = [f for ff in faces for f in ff
                    if abs(f[0] + f[2] / 2 - cx) < fw * 0.04 and abs(f[1] + f[3] / 2 - cy) < fh * 0.06
                    and 0.7 < f[2] / w < 1.4]
            if len(hits) > best_hits:
                best, best_hits = hits, len(hits)
    if not best or best_hits < len(faces) * 0.5:
        return None
    x = statistics.median(f[0] for f in best)
    y = statistics.median(f[1] for f in best)
    w = statistics.median(f[2] for f in best)
    h = statistics.median(f[3] for f in best)
    # La cámara es más ancha que la cara. Cuanto más amplio el recorte, menos se amplía y mejor calidad:
    # unas 4,5 caras de ancho (y nunca menos de un tercio del vídeo), con la proporción del piso de arriba.
    cw = min(fw, max(w * 4.5, fw / 3))
    ch = min(fh, cw * TOP_H / p.W)
    cx = min(max(0, x + w / 2 - cw / 2), fw - cw)
    cy = min(max(0, y + h * 0.45 - ch * 0.55), fh - ch)  # cara algo baja: el gancho va encima
    even = lambda v: int(v) // 2 * 2
    return even(cx), even(cy), even(cw), even(ch), fw, fh


def render_facecam(video, ass_file, dest, cam):
    cx, cy, cw, ch, fw, fh = cam
    gw = min(fw, int(fh * p.W / BOT_H) // 2 * 2)   # recorte del centro del juego con la proporción del piso de abajo
    gx = (fw - gw) // 2
    fc = (f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={p.W}:{TOP_H}:force_original_aspect_ratio=increase:flags=lanczos,crop={p.W}:{TOP_H},unsharp=5:5:0.8[top];"
          f"[0:v]crop={gw}:{fh}:{gx}:0,scale={p.W}:{BOT_H}:force_original_aspect_ratio=increase,crop={p.W}:{BOT_H}[bot];"
          f"[top][bot]vstack,ass={ass_file.name}:fontsdir={p.FONTS_DIR}[v]")
    p.run(["ffmpeg", "-y", "-i", str(video), "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-r", "30",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dest)], cwd=ass_file.parent)


def build(clip, hook, cfg):
    """Descarga, transcribe y monta el clip en formato cara arriba. Devuelve (ruta, None) o (None, motivo)."""
    brand = cfg["brand"]
    login = clip.get("broadcaster_login") or clip["broadcaster_name"].lower()
    work = p.WORK_DIR / ("fc_" + clip["id"])
    work.mkdir(parents=True, exist_ok=True)
    src, audio = work / "src.mp4", work / "audio.mp3"
    p.download(clip["url"], src)
    cam = detect_cam(src)
    if not cam:
        return None, "no se ve una cámara con cara fija"
    p.extract_audio(src, audio)
    lang = (clip.get("language") or "es").split("-")[0]
    lang = lang if lang in p.WHISPER_LANGS else "es"
    tr = p.clean_transcript(p.groq_transcribe(audio, cfg["groq"]["transcribe_model"], lang))
    if lang != "es":
        tr = {"segments": p.groq_translate_segments(cfg["groq"]["llm_model"], tr.get("segments") or [], lang)}
    ass_file = work / "subs.ass"
    # Gancho arriba del todo y subtítulos en la unión de los dos pisos, que es donde se mira.
    ass_file.write_text(p.build_ass(p.words_from_transcript(tr), hook, f"{brand['handle']}  ·  twitch.tv/{login}", brand,
                                    hook_margin=60, sub_margin=p.H - TOP_H - 55), encoding="utf-8")
    p.OUT_DIR.mkdir(exist_ok=True)
    out = p.OUT_DIR / f"facecam_{login}_{clip['id'][:20]}.mp4"
    render_facecam(src, ass_file, out, cam)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--clip")
    args = ap.parse_args()
    cfg = p.yaml.safe_load((p.ROOT / "config.yaml").read_text())
    tw = p.Twitch(p.env("TWITCH_CLIENT_ID"), p.env("TWITCH_CLIENT_SECRET"))
    if args.clip:
        jobs = [(args.clip.rstrip("/").split("/")[-1].split("?")[0], None)]
    else:
        state = json.loads(p.STATE_FILE.read_text())["processed"]
        sent = sorted(((k, v) for k, v in state.items()
                       if v.get("status") == "enviado" and v.get("cat") in ("reaccion", "gracioso")),
                      key=lambda kv: -kv[1]["t"])
        jobs = [(k, v.get("hook")) for k, v in sent[:args.n]]
    ok = 0
    for cid, hook in jobs:
        data = tw.get("clips", {"id": cid})
        if not data:
            continue
        clip = data[0]
        try:
            out, why = build(clip, hook or clip["title"], cfg)
        except Exception as e:
            out, why = None, f"error: {str(e)[:200]}"
        if out:
            p.send_telegram(out, f"🧪 PRUEBA formato cara arriba · {clip['broadcaster_name']}\n{hook or ''}\n{clip['url']}")
            ok += 1
        else:
            p.send_telegram_text(f"🧪 PRUEBA cara arriba · {clip['broadcaster_name']}: {why} (se quedaría en el formato actual)\n{clip['url']}")
        p.log(f"[{clip['broadcaster_name']}] {'OK' if out else why}")
    p.send_telegram_text(f"🧪 Prueba terminada: {ok}/{len(jobs)} clips con cámara detectada.")


if __name__ == "__main__":
    main()
