# GTA6 clips (streamers hispanos)

Cada ~10 min, GitHub Actions:
1. Busca los clips de Twitch más vistos (con más vistas por hora) de los streamers de `config.yaml`, solo de GTA VI.
2. Los descarga y transcribe en español (Groq Whisper).
3. La IA los puntúa como "reacción". Descarta los aburridos y los que parecen música de la radio (copyright) y escribe el gancho.
4. Monta el vídeo 9:16 con la marca: gancho en caja de color arriba, subtítulos palabra a palabra y crédito al streamer.
5. Te lo envía a Telegram con el texto para publicar. Tú lo revisas y lo subes a TikTok, Reels y Shorts.

## Puesta en marcha (unos 15 min)

1. **Twitch:** en https://dev.twitch.tv/console/apps, "Register Your Application". OAuth Redirect: `http://localhost`. Categoría: Other. Copia el **Client ID** y genera un **Client Secret**.
2. **Groq:** en https://console.groq.com/keys, crea una **API key**.
3. **Telegram:** habla con @BotFather, envía `/newbot` y copia el **token**. Escríbele cualquier cosa a tu bot y luego abre
   `https://api.telegram.org/bot<TOKEN>/getUpdates`. El número de `"chat":{"id":...}` es tu **chat ID**.
4. **GitHub:** crea un repo **público**. Así los minutos de Actions son ilimitados, y los secretos no se ven aunque el repo sea público. Sube este directorio.
5. En el repo, ve a Settings → Secrets and variables → Actions y añade estos secretos:
   `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
6. En `config.yaml`, cambia `brand.handle` por tu @.
7. Ve a Actions → clips → "Run workflow". Ahí puedes pegar la URL de un clip concreto para probar.

## Ajustes
- `config.yaml`: streamers, vistas mínimas, clips por ejecución, nota mínima de la IA y color de marca.
- Prueba local: `python pipeline.py --dry-run` (necesita las mismas variables de entorno). Los vídeos se quedan en `out/`.
- El cron de GitHub puede retrasarse unos minutos en horas punta.

## Recopilaciones (vídeos largos para YouTube)
Los domingos a las 10:00 se montan solas tres: **bugs**, **curiosidades** y **mejores momentos** (si hay al menos 3 clips de cada una).
Para lanzar una cuando quieras: Actions → recopilacion-semanal → "Run workflow" → elige la categoría y los días.
Te llega a Telegram con el enlace de descarga, el título, la descripción y los capítulos.
