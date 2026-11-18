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

# Modelo de reserva (con su propio cupo) por si el principal devuelve 429 o está caído.
GROQ_FALLBACK = {"whisper-large-v3": "whisper-large-v3-turbo", "openai/gpt-oss-120b": "openai/gpt-oss-20b"}


class GroqBusy(Exception):
    """Groq sin cupo: el clip se reintenta en la siguiente vuelta en vez de perderse."""


def groq_call(path, model, make_kwargs):
    for m in [model] + ([GROQ_FALLBACK[model]] if model in GROQ_FALLBACK else []):
        r = requests.post(f"https://api.groq.com/openai/v1/{path}", timeout=120,
                          headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"}, **make_kwargs(m))
        if r.status_code == 429 or r.status_code >= 500:
            log(f"Groq {m}: {r.status_code}, pruebo reserva")
            continue
        r.raise_for_status()
        return r.json()
    raise GroqBusy(f"Groq sin cupo para {path}")


def groq_chat_json(model, prompt, temperature):
    data = groq_call("chat/completions", model, lambda m: {"json": {
        # reasoning_effort low: misma nota con ~1/3 de tokens, para no agotar el cupo diario de Groq
        "model": m, "temperature": temperature, "response_format": {"type": "json_object"}, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": prompt}]}})
    return json.loads(data["choices"][0]["message"]["content"])


# Frases que Whisper se inventa en silencios (vienen de subtítulos de su entrenamiento).
HALLUCINATIONS = re.compile(r"amara\.org|subt[ií]tul|legenda|suscr[ií]bete|gracias por ver|thanks for watching|"
                            r"ご視聴|字幕|продолжение следует|untertitel", re.I)


def clean_transcript(tr):
    """Quita segmentos sin voz o alucinados, y las palabras que caen dentro de ellos."""
    segs = tr.get("segments") or []
    bad = [s for s in segs if s.get("no_speech_prob", 0) > 0.6 or HALLUCINATIONS.search(s.get("text", ""))]
    if not bad:
        return tr
    inside = lambda t: any(b["start"] <= t < b["end"] for b in bad)
    good = [s for s in segs if s not in bad]
    return {**tr, "segments": good, "words": [w for w in tr.get("words") or [] if not inside(w["start"])],
            "text": " ".join(s["text"].strip() for s in good)}


def groq_transcribe(audio_path, model, language="es"):
    audio = audio_path.read_bytes()
    return groq_call("audio/transcriptions", model, lambda m: {
        "files": {"file": (audio_path.name, audio, "audio/mpeg")},
        "data": [("model", m), ("language", language), ("response_format", "verbose_json"),
                 ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]})


PROMPT = """Eres el editor de una cuenta de TikTok/Reels/Shorts en español sobre streamers jugando a GTA 6: \
reacciones, bugs, curiosidades del juego, momentos épicos y graciosos.

Clip de Twitch de {streamer} (idioma del directo: {lang}). Título que le puso el chat: "{title}". Vistas: {views}.
Si el streamer no es hispanohablante, el vídeo irá subtitulado al español y el gancho (en español) debe presentarlo \
para el público hispano: si es muy famoso, por su nombre ("Kai Cenat, el streamer más visto de EEUU, ..."); \
si no, por su nacionalidad según el idioma ("Streamer japonés reacciona a...", "Así reacciona Corea a GTA 6...").
Transcripción (puede tener errores):
\"\"\"{transcript}\"\"\"

Decide si merece publicarse. Tú solo ves la transcripción, no la imagen: muchas vistas significa que el chat \
lo encontró gracioso o épico, así que NO lo descartes solo por tener poca voz o poco contexto. \
Descarta solo si es claramente charla aburrida, o si la transcripción es mayormente letra de una canción \
(riesgo de copyright por la radio del juego).

Clasifica también el clip:
- "bug": un fallo del juego (físicas raras, coches volando, NPCs atravesando paredes, personajes atascados...); \
se nota porque el streamer dice "bug", "glitch", "¿qué ha pasado?", "el juego está roto", o lo dice el título.
- "curiosidad": el streamer descubre un detalle del juego, un easter egg, una mecánica sorprendente o realista \
("mira qué detalle", "¿esto se puede hacer?", "no sabía que...").
- "reaccion": rabia, gritos, flipadas, emoción del streamer.
- "gracioso": situación cómica que no es un bug.
- "epico": jugada, persecución o momento espectacular.

Responde SOLO con JSON:
{{"publicar": true/false,
  "puntuacion": 1-10,
  "categoria": "bug" | "curiosidad" | "reaccion" | "gracioso" | "epico" | "otro",
  "gancho": "texto corto (máx 55 caracteres) para poner arriba del vídeo, que genere curiosidad, menciona al streamer",
  "descripcion": "1-2 frases para el post, natural, en español",
  "musica": true/false (true si en la transcripción aparece algún fragmento de letra de canción, o el streamer \
habla de la radio/una canción que suena: riesgo de que TikTok silencie el vídeo o YouTube lo reclame),
  "motivo": "por qué (breve)"}}"""


def groq_translate_segments(model, segments, lang):
    """Traduce al español el texto de cada segmento manteniendo sus tiempos."""
    src = [{"i": i, "t": seg["text"].strip()} for i, seg in enumerate(segments)]
    prompt = (f"Traduce a español neutro, coloquial y natural (es un streamer hablando, idioma original: {lang}), "
              "que se entienda igual en España y en Latinoamérica: evita palabras que en algún país son vulgares "
              "(p. ej. 'coger': usa 'agarrar', 'tomar', 'pillar'...). "
              "Mantén insultos/exclamaciones con su intensidad. "
              'Responde SOLO con JSON: {"es": ["traducción 0", "traducción 1", ...]} con el mismo número de elementos.\n'
              + json.dumps(src, ensure_ascii=False))
    for _ in range(2):  # a veces junta o parte un segmento: se repite antes de dar el clip por fallido
        es = groq_chat_json(model, prompt, 0.2).get("es", [])
        if len(es) == len(segments):
            break
    else:
        raise RuntimeError(f"traducción incompleta ({len(es)}/{len(segments)} segmentos)")
    return [{**seg, "text": t} for seg, t in zip(segments, es)]


def groq_judge(model, streamer, title, views, transcript, lang="es"):
    return groq_chat_json(model, PROMPT.format(
        streamer=streamer, title=title, views=views, lang=lang, transcript=transcript[:3000] or "(sin voz)"), 0.4)


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


# Idiomas que Whisper transcribe bien (código de idioma de Twitch).
WHISPER_LANGS = {"es", "en", "pt", "fr", "de", "it", "ja", "ko", "zh", "ru", "pl", "tr", "ar", "nl", "sv", "uk"}

CAT_LABEL = {"bug": "🐛 BUG · ", "curiosidad": "🔍 CURIOSIDAD · ", "reaccion": "😱 REACCIÓN · ",
             "gracioso": "😂 GRACIOSO · ", "epico": "🔥 ÉPICO · "}


def pending(state, clip_id):
    """Clip aún por procesar (nuevo, o con error y menos de 3 intentos)."""
    prev = state["processed"].get(clip_id)
    return prev is None or (prev.get("status") == "error" and prev.get("tries", 1) < 3)


def sent_last_hour(state):
    return sum(1 for v in state["processed"].values()
               if v.get("status") == "enviado" and v.get("t", 0) > time.time() - 3600)


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
        lang = (clip.get("language") or "es").split("-")[0]
        if lang not in WHISPER_LANGS:
            lang = "es"
        tr = clean_transcript(groq_transcribe(audio, cfg["groq"]["transcribe_model"], lang))
        transcript = (tr.get("text") or "").strip()

        verdict = groq_judge(cfg["groq"]["llm_model"], streamer, clip.get("title", ""),
                             clip.get("view_count", 0), transcript, lang)
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
        if lang != "es":
            # Subtítulos traducidos: por segmento (los tiempos por palabra no sirven al cambiar de idioma).
            tr = {"segments": groq_translate_segments(cfg["groq"]["llm_model"], tr.get("segments") or [], lang)}
        ass_file.write_text(build_ass(words_from_transcript(tr), hook, credit, brand), encoding="utf-8")
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"{datetime.now():%m%d_%H%M}_{login}_{clip['id'][:20]}.mp4"
        render(src, ass_file, out)
        log(f"[{streamer}] renderizado {out.name}")

        tags = " ".join(brand["hashtags"] + [f"#{login}"])
        warn = "⚠️ POSIBLE MÚSICA CON COPYRIGHT: revísalo antes de subir\n" if verdict.get("musica") else ""
        caption = (warn + CAT_LABEL.get(cat, "") + (f"🌍 {lang.upper()}→ES · " if lang != "es" else "") + f"⭐ {score}/10 · {streamer} · {clip.get('view_count', 0)} vistas en Twitch\n\n"
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
        now = datetime.now(timezone.utc)
        utc = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        if sel.get("global_until") and games and utc(sel["global_from"]) <= now < utc(sel["global_until"]):
            # Antes de que el juego llegue a España: top clips de GTA VI de cualquier streamer (subtitulados al español).
            log("modo global: clips de cualquier streamer")
            # Solo clips desde el desbloqueo: los de la cuenta atrás no sirven y gastarían el tope por hora.
            gsince = max(since, utc(sel["global_from"]))
            # Twitch tarda minutos en contar las vistas de un clip nuevo: mientras tanto, vale el tamaño del directo
            # (espectadores >= stream_viewers_factor x vistas mínimas del idioma).
            factor = sel.get("stream_viewers_factor")
            viewers = {}
            if factor:
                for gid in games:
                    viewers.update({s["user_id"]: s["viewer_count"] for s in tw.get("streams", {"game_id": gid, "first": 100})})
            gclips = [c for gid in games for c in tw.get("clips", {"game_id": gid, "first": 100,
                                                                 "started_at": gsince.strftime("%Y-%m-%dT%H:%M:%SZ")})]
            # Los clips con pocas vistas no salen en el top 100 del juego: a esos streamers se les pide aparte.
            got = {c["id"] for c in gclips}
            for login in sel.get("streamer_min_views", {}):
                if login in ids:
                    gclips += [c for c in tw.clips(ids[login], gsince) if c["id"] not in got and c.get("game_id") in games]
            for c in gclips:
                c["broadcaster_login"] = re.sub(r"[^a-z0-9_]", "", c["broadcaster_name"].lower()) or c["broadcaster_id"]
                lang = c.get("language", "").split("-")[0]
                if not (pending(state, c["id"]) and lang in sel["global_min_views"]
                        and c["duration"] >= sel["min_duration"]):
                    continue
                # Vistas mínimas propias para algunos streamers (sus clips apenas suman vistas durante el directo)
                c["min_views"] = sel.get("streamer_min_views", {}).get(c["broadcaster_login"], sel["global_min_views"][lang])
                if c["view_count"] < c["min_views"]:
                    live = viewers.get(c["broadcaster_id"], 0)
                    # Solo clips muy recientes: si no, la estimación tapa a los que ya tienen vistas reales.
                    if not factor or live < factor * c["min_views"] or now - utc(c["created_at"]) > timedelta(minutes=8):
                        continue
                    c["view_count"] = max(c["view_count"], live // factor)  # estimación para ordenar
                candidates.append(c)
            ids = {}
        for login, uid in ids.items():
            try:
                for c in tw.clips(uid, since):
                    c["broadcaster_login"] = login
                    if (pending(state, c["id"]) and c["view_count"] >= sel["min_views"]
                            and c["duration"] >= sel["min_duration"]
                            and (not games or c.get("game_id") in games)):
                        candidates.append(c)
            except requests.HTTPError as e:
                log(f"[{login}] error de Twitch: {e}")
        # Prioriza clips con muchas vistas para su antigüedad (lo que está explotando ahora).

        def velocity(c):
            age_h = (now - datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))).total_seconds() / 3600
            # Relativo al umbral de su idioma, para que un streamer japonés compita con uno de EEUU.
            boost = sel.get("boost", {}).get(c["broadcaster_login"].lower(), 1)  # streamers a priorizar
            return boost * c["view_count"] / c.get("min_views", 1) / max(age_h, 0.25)
        candidates.sort(key=velocity, reverse=True)
        # Tope por streamer: que uno muy activo no se coma el cupo por hora entero.
        caps = sel.get("streamer_max_per_hour", {})
        if caps:
            used = {}
            for v in state["processed"].values():
                if v.get("status") == "enviado" and v.get("t", 0) > time.time() - 3600:
                    used[v.get("login", "")] = used.get(v.get("login", ""), 0) + 1
            kept = []
            for c in candidates:
                login = c["broadcaster_login"]
                if login in caps and used.get(login, 0) >= caps[login]:
                    continue
                used[login] = used.get(login, 0) + 1
                kept.append(c)
            candidates = kept
        room = max(0, sel["max_per_hour"] - sent_last_hour(state))
        if room == 0:
            log(f"tope de {sel['max_per_hour']} vídeos/hora alcanzado; espero")
        candidates = candidates[:min(sel["max_per_run"], room)]

    log(f"{len(candidates)} clips a procesar")
    for c in candidates:
        try:
            result = process(c, cfg, args.dry_run)
        except GroqBusy as e:  # sin cupo de IA: no se marca, se reintenta en la siguiente vuelta
            log(f"{e}; paro esta vuelta")
            break
        except Exception as e:  # un clip roto no debe parar al resto
            log(f"[{c['broadcaster_name']}] ERROR: {e}")
            tries = state["processed"].get(c["id"], {}).get("tries", 0) + 1
            result = {"status": "error", "tries": tries, "error": str(e)[:200]}
        state["processed"][c["id"]] = {"t": time.time(), "streamer": c["broadcaster_name"], **result}

    if not args.dry_run:
        STATE_FILE.write_text(json.dumps(state, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
