#!/usr/bin/env python3
"""Pipeline de clips: Twitch top clips -> transcripción -> filtro IA -> vídeo 9:16 con marca -> Telegram.

Uso:
  python pipeline.py                 # ejecución normal (lo que corre GitHub Actions)
  python pipeline.py --dry-run       # no envía a Telegram ni guarda estado; deja los vídeos en out/
  python pipeline.py --clip URL      # procesa un clip concreto de Twitch (ignora filtros)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "state.json"
FONTS_DIR = ROOT / "fonts"
OUT_DIR = ROOT / "out"
WORK_DIR = ROOT / "work"

W, H = 1080, 1920
FG_H = 608              # alto del vídeo original escalado a 1080 de ancho (16:9)
FG_Y = (H - FG_H) // 2 - 80


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"Falta la variable de entorno {name}")
    return v


# ---------------------------------------------------------------- Twitch

class Twitch:
    def __init__(self, client_id, client_secret):
        r = requests.post("https://id.twitch.tv/oauth2/token", params={
            "client_id": client_id, "client_secret": client_secret,
            "grant_type": "client_credentials"}, timeout=20)
        r.raise_for_status()
        self.h = {"Client-ID": client_id, "Authorization": f"Bearer {r.json()['access_token']}"}

    def get(self, path, params):
        r = requests.get(f"https://api.twitch.tv/helix/{path}", headers=self.h, params=params, timeout=20)
        r.raise_for_status()
        return r.json()["data"]

    def user_ids(self, logins):
        data = self.get("users", [("login", l) for l in logins])
        found = {u["login"].lower(): u["id"] for u in data}
        for l in logins:
            if l.lower() not in found:
                log(f"AVISO: el login de Twitch '{l}' no existe")
        return found

    def game_ids(self, names):
        if not names:
            return set()
        data = self.get("games", [("name", n) for n in names])
        if not data:
            log(f"AVISO: no se encontró ningún juego {names}; no se filtra por juego")
        return {g["id"] for g in data}

    def clips(self, broadcaster_id, since):
        return self.get("clips", {"broadcaster_id": broadcaster_id, "first": 20,
                                  "started_at": since.strftime("%Y-%m-%dT%H:%M:%SZ")})


# ---------------------------------------------------------------- Groq

def groq_transcribe(audio_path, model):
    with open(audio_path, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"},
            files={"file": (audio_path.name, f, "audio/mpeg")},
            data=[("model", model), ("language", "es"), ("response_format", "verbose_json"),
                  ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")],
            timeout=120)
    r.raise_for_status()
    return r.json()


PROMPT = """Eres el editor de una cuenta de TikTok/Reels/Shorts en español especializada en REACCIONES de streamers \
(rabia, gritos, flipadas, risas, momentos épicos) jugando a GTA 6.

Clip de Twitch de {streamer}. Título que le puso el chat: "{title}". Vistas: {views}.
Transcripción (puede tener errores):
\"\"\"{transcript}\"\"\"

Decide si merece publicarse. Tú solo ves la transcripción, no la imagen: muchas vistas significa que el chat \
lo encontró gracioso o épico, así que NO lo descartes solo por tener poca voz o poco contexto. \
Descarta solo si es claramente charla aburrida, o si la transcripción es mayormente letra de una canción \
(riesgo de copyright por la radio del juego).

Clasifica también el clip. "bug" = un fallo del juego (físicas raras, coches volando, NPCs atravesando paredes, \
personajes atascados...); se nota porque el streamer dice cosas como "bug", "glitch", "¿qué ha pasado?", \
"el juego está roto", o porque lo dice el título.

Responde SOLO con JSON:
{{"publicar": true/false,
  "puntuacion": 1-10,
  "categoria": "bug" | "reaccion" | "epico" | "otro",
  "gancho": "texto corto (máx 55 caracteres) para poner arriba del vídeo, que genere curiosidad, menciona al streamer",
  "descripcion": "1-2 frases para el post, natural, en español",
  "motivo": "por qué (breve)"}}"""


def groq_judge(model, streamer, title, views, transcript):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"},
        json={"model": model, "temperature": 0.4, "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": PROMPT.format(
                  streamer=streamer, title=title, views=views, transcript=transcript[:3000] or "(sin voz)")}]},
        timeout=60)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------- Subtítulos ASS

def ass_time(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass_color(rgb_hex, alpha="00"):
    r, g, b = rgb_hex[0:2], rgb_hex[2:4], rgb_hex[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()


def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def chunk_words(words, max_words=3, max_gap=0.6):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > max_gap):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,]$", w["word"]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def words_from_transcript(tr):
    words = [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
             for w in tr.get("words") or [] if w.get("word", "").strip()]
    if words:
        return words
    # Sin marcas por palabra: repartir el tiempo de cada segmento entre sus palabras.
    for seg in tr.get("segments") or []:
        toks = seg["text"].split()
        if not toks:
            continue
        step = (seg["end"] - seg["start"]) / len(toks)
        words += [{"word": t, "start": seg["start"] + i * step, "end": seg["start"] + (i + 1) * step}
                  for i, t in enumerate(toks)]
    return words


def build_ass(words, hook, credit, brand):
    color, font = brand["color"], brand["font"]
    white, black = "&H00FFFFFF", "&H00000000"
    sub_margin = H - (FG_Y + FG_H) - 260   # subtítulos justo debajo del vídeo
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: Hook,{font},78,{black},{black},{ass_color(color)},{ass_color(color)},0,0,0,0,100,100,0,0,3,18,0,8,90,90,{FG_Y - 330},1",
        f"Style: Sub,{font},96,{white},{white},{black},&H80000000,0,0,0,0,100,100,1,0,1,7,3,2,70,70,{sub_margin},1",
        f"Style: Credit,{font},44,&H40FFFFFF,&H40FFFFFF,&H80000000,&H00000000,0,0,0,0,100,100,1,0,1,3,0,2,60,60,170,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 0,{ass_time(0)},{ass_time(600)},Hook,,0,0,0,,{ass_escape(hook)}",
        f"Dialogue: 0,{ass_time(0)},{ass_time(600)},Credit,,0,0,0,,{ass_escape(credit)}",
    ]
    hi = ass_color(color)
    chunks = chunk_words(words)
    for ci, chunk in enumerate(chunks):
        chunk_end = chunk[-1]["end"] + 0.25
        if ci + 1 < len(chunks):
            chunk_end = min(chunk_end, chunks[ci + 1][0]["start"])
        for wi, w in enumerate(chunk):
            start = w["start"]
            end = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else chunk_end
            if end <= start:
                continue
            text = " ".join(
                (f"{{\\c{hi}}}{ass_escape(x['word'].upper())}{{\\c{white}}}" if j == wi else ass_escape(x["word"].upper()))
                for j, x in enumerate(chunk))
            lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Sub,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- Vídeo

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} falló:\n{p.stderr[-2000:]}")
    return p


def download(url, dest):
    run(["yt-dlp", "-q", "--no-warnings", "-f", "best[ext=mp4]/best", "-o", str(dest), url])


def extract_audio(video, dest):
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dest)])


def render(video, ass_file, dest):
    fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:2,"
          f"eq=brightness=-0.18[bg];"
          f"[0:v]scale={W}:{FG_H}:force_original_aspect_ratio=decrease,pad={W}:{FG_H}:(ow-iw)/2:(oh-ih)/2[fg];"
          f"[bg][fg]overlay=0:{FG_Y},ass={ass_file.name}:fontsdir={FONTS_DIR}[v]")
    run(["ffmpeg", "-y", "-i", str(video), "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-r", "30",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dest)], cwd=ass_file.parent)


# ---------------------------------------------------------------- Telegram

def send_telegram(video, caption):
    token, chat = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    with open(video, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendVideo",
                          data={"chat_id": chat, "caption": caption[:1024], "supports_streaming": "true"},
                          files={"video": (video.name, f, "video/mp4")}, timeout=300)
    if not r.ok:
        raise RuntimeError(f"Telegram: {r.status_code} {r.text[:300]}")


def send_telegram_text(text):
    token, chat = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    for i in range(0, len(text), 4000):
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text[i:i + 4000]}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"Telegram: {r.status_code} {r.text[:300]}")


# ---------------------------------------------------------------- Orquestación

def load_state():
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"processed": {}}
    cutoff = time.time() - 14 * 86400  # 14 días: la recopilación semanal tira de aquí
    state["processed"] = {k: v for k, v in state["processed"].items() if v.get("t", 0) > cutoff}
    return state


def process(clip, cfg, dry_run):
    brand = cfg["brand"]
    streamer = clip["broadcaster_name"]
    login = clip.get("broadcaster_login") or streamer.lower()
    work = WORK_DIR / clip["id"]
    work.mkdir(parents=True, exist_ok=True)
    try:
        src, audio = work / "src.mp4", work / "audio.mp3"
        log(f"[{streamer}] descargando {clip['url']}")
        download(clip["url"], src)
        extract_audio(src, audio)
        tr = groq_transcribe(audio, cfg["groq"]["transcribe_model"])
        transcript = (tr.get("text") or "").strip()

        verdict = groq_judge(cfg["groq"]["llm_model"], streamer, clip.get("title", ""),
                             clip.get("view_count", 0), transcript)
        score = int(verdict.get("puntuacion", 0))
        cat = verdict.get("categoria", "otro")
        hook = verdict.get("gancho") or clip.get("title", "")
        meta = {"score": score, "cat": cat, "hook": hook, "url": clip["url"], "login": login,
                "views": clip.get("view_count", 0), "duration": clip.get("duration", 0)}
        log(f"[{streamer}] IA: {cat} publicar={verdict.get('publicar')} nota={score} · {verdict.get('motivo', '')}")
        if not clip.get("forced") and (not verdict.get("publicar") or score < cfg["selection"]["min_score"]):
            return {"status": "descartado", **meta}

        credit = f"{brand['handle']}  ·  twitch.tv/{login}"
        ass_file = work / "subs.ass"
        ass_file.write_text(build_ass(words_from_transcript(tr), hook, credit, brand), encoding="utf-8")
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"{datetime.now():%m%d_%H%M}_{login}_{clip['id'][:20]}.mp4"
        render(src, ass_file, out)
        log(f"[{streamer}] renderizado {out.name}")

        tags = " ".join(brand["hashtags"] + [f"#{login}"])
        caption = (("🐛 BUG · " if cat == "bug" else "") + f"⭐ {score}/10 · {streamer} · {clip.get('view_count', 0)} vistas en Twitch\n\n"
                   f"📋 TEXTO PARA PUBLICAR:\n{hook}\n\n{verdict.get('descripcion', '')}\n"
                   f"🎥 Créditos: twitch.tv/{login}\n{tags}\n\n🔗 Original: {clip['url']}")
        if dry_run:
            log("--- caption ---\n" + caption)
        else:
            send_telegram(out, caption)
            out.unlink()
        return {"status": "enviado", **meta}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clip", help="URL de un clip de Twitch concreto")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    sel = cfg["selection"]
    tw = Twitch(env("TWITCH_CLIENT_ID"), env("TWITCH_CLIENT_SECRET"))
    state = load_state()

    if args.clip:
        slug = args.clip.rstrip("/").split("/")[-1].split("?")[0]
        data = tw.get("clips", {"id": slug})
        if not data:
            sys.exit("Clip no encontrado")
        data[0]["forced"] = True
        candidates = data
    else:
        ids = tw.user_ids(cfg["streamers"])
        games = tw.game_ids(cfg.get("games", []))
        since = datetime.now(timezone.utc) - timedelta(hours=sel["window_hours"])
        candidates = []
        for login, uid in ids.items():
            try:
                for c in tw.clips(uid, since):
                    c["broadcaster_login"] = login
                    if (c["id"] not in state["processed"] and c["view_count"] >= sel["min_views"]
                            and c["duration"] >= sel["min_duration"]
                            and (not games or c.get("game_id") in games)):
                        candidates.append(c)
            except requests.HTTPError as e:
                log(f"[{login}] error de Twitch: {e}")
        # Prioriza clips con muchas vistas para su antigüedad (lo que está explotando ahora).
        now = datetime.now(timezone.utc)

        def velocity(c):
            age_h = (now - datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))).total_seconds() / 3600
            return c["view_count"] / max(age_h, 0.25)
        candidates.sort(key=velocity, reverse=True)
        candidates = candidates[:sel["max_per_run"]]

    log(f"{len(candidates)} clips a procesar")
    for c in candidates:
        try:
            result = process(c, cfg, args.dry_run)
        except Exception as e:  # un clip roto no debe parar al resto
            log(f"[{c['broadcaster_name']}] ERROR: {e}")
            result = {"status": "error", "error": str(e)[:200]}
        state["processed"][c["id"]] = {"t": time.time(), "streamer": c["broadcaster_name"], **result}

    if not args.dry_run:
        STATE_FILE.write_text(json.dumps(state, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
