# src/music_openverse.py
import os
import random
import time
import requests, json
from pathlib import Path
from urllib.parse import urlparse



OPENVERSE_CLIENT_ID     = os.getenv("OPENVERSE_CLIENT_ID")
OPENVERSE_CLIENT_SECRET = os.getenv("OPENVERSE_CLIENT_SECRET")

TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"
AUDIO_URL = "https://api.openverse.org/v1/audio/"

UA = "ShortsAuto/1.0 (+https://example.local) Python-requests"
TOKEN_CACHE = Path.home() / ".openverse_token.json"

# ---------------------------
# OAuth2 Client Credentials
# ---------------------------

def openverse_auth_token():
    global OPENVERSE_TOKEN
    if OPENVERSE_TOKEN:  # si ya tenemos token en memoria, usarlo
        return OPENVERSE_TOKEN
    r = requests.post(
        "https://api.openverse.org/v1/auth_tokens/token/",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": OPENVERSE_CLIENT_ID,
            "client_secret": OPENVERSE_CLIENT_SECRET,
        },
        timeout=20,
    )
    r.raise_for_status()
    OPENVERSE_TOKEN = r.json()["access_token"]
    print(F"[openverse] TOKEN {OPENVERSE_TOKEN}")
    return OPENVERSE_TOKEN


def openverse_search_synthwave(q="synthwave", page_size=30, sources="jamendo"):
    token = openverse_auth_token()
    params = {
        "q": q,
        "license_type": "commercial",                  # sólo licencias aptas p/uso comercial
        "source": sources,                             # buenas fuentes musicales
        "page_size": page_size,
        "fields": "title,creator,license,url,duration,foreign_landing_url,source",
    }
    print("[INFO][openverse] GET https://api.openverse.org/v1/audio/", "params=", params)
    r = requests.get(
        "https://api.openverse.org/v1/audio/",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []

    # nos quedamos con ítems que traen un URL directo reproducible (mp3/ogg/wav)
    def is_playable(item):
        u = item.get("url") or ""
        if not u.startswith("http"):
            return False
        ext = os.path.splitext(urlparse(u).path)[1].lower()
        return ext in (".mp3", ".ogg", ".wav", ".flac", ".m4a")

    playable = [it for it in results if is_playable(it)]
    print(f"[openverse] candidatos reproducibles: {len(playable)}")
    return playable

def pick_and_download_openverse(queries=("synthwave","retrowave","outrun","80s electronic","chiptune 80s"),
                                out="music.mp3"):
    last_err = None
    for q in queries:
        try:
            items = openverse_search_synthwave(q=q)
            if not items:
                continue
            choice = random.choice(items)
            url = choice["url"]
            print("[openverse] elegido:", choice.get("title"), "-", choice.get("creator"), url)
            # descargar con stream
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1<<20):
                        if chunk:
                            f.write(chunk)
            meta = {
                "title": choice.get("title"),
                "creator": choice.get("creator"),
                "license": choice.get("license"),
                "source": choice.get("source"),
                "landing": choice.get("foreign_landing_url"),
                "url": url,
            }
            return out, meta
        except Exception as e:
            print("[openverse] fallo descarga para", q, "->", e)
            last_err = e
    raise last_err or RuntimeError("No se pudo descargar música desde Openverse")

def mix_music_into_video(video_in="tmp_base.mp4", music="music.mp3", out="short_with_music.mp4",
                         music_db=-18, ducking_db=-6):
    """
    - Normaliza música a ~-18 LUFS y la baja unos dB (ducking simple)
    - Mantiene el audio original (voz) del video.
    """
    # baja volumen de música
    vfilt = f"[1:a]loudnorm=I={music_db}:TP=-1.5:LRA=11:print_format=summary,volume={ducking_db}dB[bg]"
    # mezcla voz (0:a) + bg -> outa
    # usa amix con pesos (voz 1.0, bg 0.6 por ejemplo)
    cmd = (
        f'ffmpeg -y -i "{video_in}" -i "{music}" '
        f'-filter_complex "{vfilt};[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=1.0[outa]" '
        f'-map 0:v -map "[outa]" -c:v copy -c:a aac -b:a 192k "{out}"'
    )
    print(">>", cmd)
    import subprocess; subprocess.run(cmd, shell=True, check=True)
    return out

# //////////////////////////

def _save_token(tok: dict):
    TOKEN_CACHE.write_text(json.dumps(tok, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_token():
    if TOKEN_CACHE.exists():
        try:
            return json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _request_new_token():
    if not OPENVERSE_CLIENT_ID or not OPENVERSE_CLIENT_SECRET:
        raise RuntimeError("Faltan OPENVERSE_CLIENT_ID / OPENVERSE_CLIENT_SECRET")
    headers = {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA}
    data = {
        "grant_type": "client_credentials",
        "client_id": OPENVERSE_CLIENT_ID,
        "client_secret": OPENVERSE_CLIENT_SECRET,
    }
    r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=20)
    r.raise_for_status()
    resp = r.json()
    # guardamos cuándo expira (epoch segundos)
    now = int(time.time())
    resp["_obtained_at"] = now
    resp["_expires_at"] = now + int(resp.get("expires_in", 0)) - 60  # 60s de margen
    _save_token(resp)
    return resp

def get_openverse_token():
    tok = _load_token()
    now = int(time.time())
    if not tok or now >= int(tok.get("_expires_at", 0)):
        tok = _request_new_token()
    return tok["access_token"]

def ov_headers():
    token = get_openverse_token()
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": UA,
        "Accept": "application/json",
    }

# ---------------------------
# Búsqueda y descarga (audio)
# ---------------------------
def openverse_search_audio(
    q="synthwave retrowave 80s",
    page_size=30,
    sources="jamendo",
    license_type="commercial",  # reutilizable comercialmente
    fields="title,creator,license,files,duration,foreign_landing_url",
    retries=1,
):
    params = {
        "q": q,
        "license_type": license_type,
        "source": sources,
        "page_size": page_size,
        "fields": fields,
    }
    last = None
    for _ in range(retries + 1):
        try:
            print("[openverse] GET", AUDIO_URL, "params=", params)
            r = requests.get(AUDIO_URL, headers=ov_headers(), params=params, timeout=20)
            if r.status_code == 401:
                # token pudo expirar “antes de tiempo”: forzamos refresh
                _request_new_token()
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            last = e
            time.sleep(0.8)
    raise last or RuntimeError("Openverse search failed")

def pick_and_download_openverse(
    q_list=None,
    out="music.mp3",
    min_dur=20,
    max_dur=90,
):
    """
    Elige al azar entre varias queries. Prefiere resultados con duración 20–90s.
    Devuelve (ruta_mp3, metadatos_dict).
    """
    q_list = q_list or [
        "synthwave",
        "retrowave",
        "80s electronic",
        "outrun",
        "chiptune 80s",
    ]
    random.shuffle(q_list)
    last_err = None
    for q in q_list:
        try:
            results = openverse_search_audio(q=q)
            candidates = []
            for it in results:
                dur = it.get("duration") or 0
                files = it.get("files") or []
                # buscar archivo reproducible (mp3 o similar)
                mp3s = [f for f in files if "mp3" in (f.get("filetype","").lower())]
                if min_dur <= dur <= max_dur and mp3s:
                    mp3s.sort(key=lambda f: f.get("bitrate", 0) or 0, reverse=True)
                    candidates.append((it, mp3s[0]["url"]))
            if not candidates:
                continue
            meta, url = random.choice(candidates)
            print(f"[openverse] elegido: {meta.get('title')} – {meta.get('creator')} :: {url}")
            with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        if chunk: f.write(chunk)
            # devolvemos metadatos útiles (p/atribución si hiciera falta)
            meta_out = {
                "title": meta.get("title"),
                "creator": meta.get("creator"),
                "license": meta.get("license"),
                "landing": meta.get("foreign_landing_url"),
                "source_url": url,
                "duration": meta.get("duration"),
            }
            Path(out).with_suffix(".json").write_text(
                json.dumps(meta_out, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            return out, meta_out
        except requests.HTTPError as e:
            last_err = e
            print("[openverse] HTTPError:", e)
        except Exception as e:
            last_err = e
            print("[openverse] error:", e)
    raise last_err or RuntimeError("No se pudo descargar música desde Openverse")