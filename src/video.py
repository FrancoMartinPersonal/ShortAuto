from pathlib import Path
import os, re, math, random, unicodedata,datetime, pprint
import requests, time
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, concatenate_videoclips, vfx
from moviepy.editor import ImageClip  # <-- NUEVO
from glob import glob                 # <-- NUEVO
from sentence_transformers import SentenceTransformer, util
from src.image_ai import generate_image_hf



# --- DEBUG LOG ---

DEBUG = True  # ponelo en False si no querés ruido en consola
pp = pprint.PrettyPrinter(indent=2, width=120, compact=True)

def dlog(*args):
    if DEBUG:
        print(*args)

AUDIT = {
    "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "scenes": []
}

LOCAL_ASSETS = glob("assets/*.mp4")   # fallback de clips locales (opcional)


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# --- CONFIG ---
W, H = 1080, 1920
FPS = 30
BITRATE = "6000k"
BG_COLOR = (0,0,0)  # fallback si no hay b-roll
SEARCH_PER_SEG = 1  # 1 clip por segmento

_EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

STOP_ES = set("""
a al algo algunas algunos ante antes aquel aquella aquellas aquellos 
aquí así aún cada casi como con contra cual cuales cuando de del 
desde donde dos el él ella ellas ellos en entre era eran es esa 
esas ese eso esos esta estaba estaban estás este esto estos fin
 fue fueron ha haber había habian han hasta hay la las le les lo
  los más mas me mi mis mucha muchos muy nada ni no nos nosotras 
  nosotros o os otra otras otro otros para pero poco por porque 
  qué que quien quienes se sin sobre su sus tal también tanto te
   tener tiene tienen toda todas todo todos tras tu tus un una 
   uno unos vuestra vuestras vuestro vuestros y ya
""".split())

load_dotenv()
PEXELS_KEY = os.getenv("PEXELS_API_KEY")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _candidate_words(text: str, max_words=30):
    t = _strip_accents(text.lower())
    words = re.findall(r"[a-záéíóúñ]+", t)
    words = [w for w in words if len(w) > 3 and w not in STOP_ES]
    # dedup preservando orden
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= max_words:
            break
    return out

def visual_keywords(text: str, top_k=3):
    """Top-k palabras más cercanas al embedding de la frase."""
    words = _candidate_words(text)
    if not words:
        return [text.strip()][:top_k]
    emb_text = _EMB_MODEL.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    emb_words = _EMB_MODEL.encode(words, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb_text, emb_words)[0]  # (N,)
    idxs = sims.argsort(descending=True)[:top_k].tolist()
    return [words[i] for i in idxs]

def build_queries_for_phrase_embeddings(text: str, top_k=3, max_out=8):
    """Arma queries: palabras sueltas + combos cortos + fallback frase completa."""
    kws = visual_keywords(text, top_k=top_k)
    queries = []
    queries.extend(kws)                       # "sangre", "glucosa", "laboratorio"
    if len(kws) >= 2: queries.append(" ".join(kws[:2]))   # "sangre glucosa"
    if len(kws) >= 3: queries.append(" ".join(kws[:3]))   # "sangre glucosa laboratorio"
    queries.append(text.strip())              # fallback final

    seen, out = set(), []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q); out.append(q)
        if len(out) >= max_out:
            break

    # Log opcional (si querés ver qué se manda)
    print(f"[emb] kws={kws} → queries={out}")
    return out
# --- FIN NUEVO ---

def make_photo_clip(img_path, dur, zoom_end=1.08):
    # Carga, adapta a 1080x1920 y aplica zoom suave (Ken Burns)
    base = ImageClip(img_path).set_duration(dur)
    base = fit_image_vertical(base)
    # zoom de 1.00 -> 1.08 en 'dur' segundos
    kz = lambda t: 1.0 + (zoom_end - 1.0) * (t / max(dur, 1e-6))
    return base.resize(kz)

def pexels_photos_search(q, n=5):
    if not PEXELS_KEY: return []
    url = "https://api.pexels.com/v1/search"
    params = {"query": q, "per_page": n, "orientation": "portrait", "size": "large"}
    r = requests.get(url, headers={"Authorization": PEXELS_KEY}, params=params, timeout=20)
    r.raise_for_status()
    photos = r.json().get("photos", [])
    out = []
    for p in photos:
        src = p.get("src", {})
        out.append(src.get("large2x") or src.get("portrait") or src.get("original") or src.get("large"))
    return [u for u in out if u]


def pexels_search(q, n=5):  # antes n=1
    if not PEXELS_KEY: return []
    url = "https://api.pexels.com/videos/search"
    params = {"query": q, "per_page": n, "orientation": "portrait", "size": "large"}
    dlog(f"[pexels] videos query='{q}' params={params}")
    r = requests.get(url, headers={"Authorization": PEXELS_KEY}, params=params, timeout=20)
    r.raise_for_status()
    vids = r.json().get("videos", [])
    dlog(f"[pexels] videos encontrados: {len(vids)}")
    out = []
    for v in vids:
        files = [f for f in v.get("video_files", []) if f.get("file_type","").startswith("video/")]
        files.sort(key=lambda f: (f.get("height",0), f.get("bitrate",0)), reverse=True)
        for f in files:
            out.append(f["link"])
    return out  # devolvemos varias opciones


def fit_image_vertical(image_clip):
    c = image_clip.resize(height=H)
    if c.w < W:
        c = c.resize(width=W)
    x_center, y_center = c.w/2, c.h/2
    return c.crop(width=W, height=H, x_center=x_center, y_center=y_center)


def download(url, out, max_retries=3):
    last_err = None
    for _ in range(max_retries):
        try:
            # Algunos CDNs de Pexels exigen Referer/UA “de navegador”
            with requests.get(
                url,
                headers={"User-Agent": UA, "Referer": "https://www.pexels.com/"},
                stream=True,
                timeout=60,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            return out
        except requests.HTTPError as e:
            last_err = e
    raise last_err or RuntimeError(f"no se pudo descargar {url}")

def fit_vertical(clip):
    # escala y recorta a 1080x1920 manteniendo centro
    c = clip.resize(height=H)
    if c.w < W:  # si aún falta ancho, seguir desde 'c'
        c = c.resize(width=W)
    x_center, y_center = c.w/2, c.h/2
    return c.crop(width=W, height=H, x_center=x_center, y_center=y_center)

def keywords_from_text(text, k=3):
    # extractor ultra simple: palabras >3 letras, sin signos, top primeras
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", text.lower())
    stop = set("de la y el los las del con por para que como una uno unas unos en un al o a se lo su sus tus mis tus eso ese esa esto esta esto esas esos muy muy pero".split())
    words = [w for w in words if len(w)>3 and w not in stop]
    if not words: return [text]
    # devolver primeras k diferentes
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    return uniq[:k]


def build_video_from_segments(segs, audio_path="voz.mp3"):
    tmp_dir = Path("tmp_broll"); tmp_dir.mkdir(exist_ok=True)
    clips = []
    used_urls = set()   # ← SOLO para este render (videos y fotos)
    last_ok_clip = None  # para reusar si falla

    for i, s in enumerate(segs):
        dur = max(1.2, s["end"] - s["start"])  # subo mínimo a 1.2s
        queries = build_queries_for_phrase_embeddings(s["text"], top_k=3, max_out=8)

        base = None


        # === IA FIRST SCENE (sólo primera frase) ===
        if i == 0:
            try:
                ia_path = generate_image_hf(s["text"], out_path=str(tmp_dir / "ia_first.jpg"))
                base = make_photo_clip(ia_path, dur)  # tu helper para fotos → clip
                print("[ia] Imagen generada para la primera frase")
            except Exception as e:
                print("[warn] IA image failed:", e)
        # === /IA FIRST SCENE ===

        # ==== VIDEOS ====
        for q in queries:
            try:
                urls = pexels_search(q, n=5)
            except Exception as e:
                print("[warn] pexels:", e); urls = []

            # no repetir dentro del mismo render
            urls = [u for u in urls if u not in used_urls]

            for j, url in enumerate(urls):
                local = tmp_dir / f"seg{i}_{j}.mp4"
                try:
                    download(url, str(local))
                    cand = VideoFileClip(str(local))
                    cand = fit_vertical(cand)
                    if cand.duration < dur:
                        reps = math.ceil(dur / cand.duration)
                        cand = concatenate_videoclips([cand]*reps).subclip(0, dur)
                    else:
                        cand = cand.subclip(0, dur)
                    base = cand
                    used_urls.add(url)  # ← marcar como usado
                    break
                except Exception as e:
                    print(f"[warn] fallo descarga/clip ({url}): {e}")
            if base: break

        if base is None:
            photo_base = None
            for q in queries:
                try:
                    purls = pexels_photos_search(q, n=5)
                except Exception as e:
                    print("[warn] pexels photos:", e); purls = []

                purls = [u for u in purls if u not in used_urls]  # ← filtrar

                for j, purl in enumerate(purls):
                    local_img = tmp_dir / f"seg{i}_{j}.jpg"
                    try:
                        download(purl, str(local_img))
                        photo_base = make_photo_clip(str(local_img), dur)
                        used_urls.add(purl)  # ← marcar como usada
                        break
                    except Exception as e:
                        print(f"[warn] fallo foto ({purl}): {e}")
                if photo_base: break

            if photo_base:
                base = photo_base
        # Fallbacks sólidos para evitar negro
        if base is None and last_ok_clip is not None:
            # reusar el último clip válido
            cand = last_ok_clip
            if cand.duration < dur:
                reps = math.ceil(dur / cand.duration)
                cand = concatenate_videoclips([cand]*reps).subclip(0, dur)
            else:
                cand = cand.subclip(0, dur)
            base = cand
        if base is None and LOCAL_ASSETS:
            try:
                cand = VideoFileClip(random.choice(LOCAL_ASSETS))
                cand = fit_vertical(cand)
                if cand.duration < dur:
                    reps = math.ceil(dur / cand.duration)
                    cand = concatenate_videoclips([cand]*reps).subclip(0, dur)
                else:
                    cand = cand.subclip(0, dur)
                base = cand
            except Exception as e:
                print("[warn] asset local falló:", e)

        if base is None:
            base = ColorClip((W, H), color=BG_COLOR, duration=dur)  # último recurso

        last_ok_clip = base
        clips.append(base)

    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio).fx(vfx.fadein,0.1).fx(vfx.fadeout,0.1)
    video.write_videofile("tmp_base.mp4", fps=FPS, codec="libx264", audio_codec="aac", bitrate=BITRATE)
    return "tmp_base.mp4"