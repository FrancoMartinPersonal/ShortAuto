import os, io, re, time, math, json, random, urllib.request, subprocess
from pathlib import Path
import requests
import textwrap  # <-- NUEVO
from faster_whisper import WhisperModel
from src.video import build_video_from_segments
from src.music import pick_and_download_openverse, mix_music_into_video
import unicodedata
import re

LANG = "es"  # idioma whisper
WHISPER_MODEL = "medium"  # tiny/base/small/medium/large (elige según tu PC)
FONT_SIZE = 18
MARGIN_V = 120
OUTLINE = 1
SHADOW = 1


def run(cmd):
    print(">>", cmd)
    return subprocess.run(cmd, shell=True, check=True)


def file_exists(p): return Path(p).exists()


def ensure_words_srt(audio, path="voz_words.srt"):
    # Fuerza subtítulos palabra-a-palabra con tiempos reales
    # (usa GPU si tenés: device="cuda", compute_type="float16")
    return srt_words_faster_whisper(
        audio_path=audio,
        out_srt=path,
        model_size="medium",
        language="es",
        device="cpu",
        compute_type="int8"
    )


def ensure_srt(audio="voz.mp3", srt="voz.srt"):
    if file_exists(srt):
        print(f"[ok] usando SRT existente: {srt}")
        return srt
    print("[i] generando SRT con Whisper (offline)…")
    run(f'whisper "{audio}" --task transcribe --language {LANG} --model {WHISPER_MODEL} --output_format srt --output_dir .')
    if not file_exists(srt):
        raise RuntimeError("No se generó voz.srt")
    return srt


def merge_short_segments(segs, min_scene=2.0, max_scene=5.0):
    """
    Une segmentos contiguos en escenas de ~2–5 s.
    - Si la escena actual aún no llega a min_scene y al sumar el siguiente
      no supera max_scene, los fusiona.
    """
    if not segs:
        return []
    out = []
    cur = {"start": segs[0]["start"], "end": segs[0]["end"], "text": segs[0]["text"]}
    for s in segs[1:]:
        cur_dur = cur["end"] - cur["start"]
        next_dur = s["end"] - cur["start"]
        if (cur_dur < min_scene) and (next_dur <= max_scene):
            cur["end"] = s["end"]
            cur["text"] = (cur["text"] + " " + s["text"]).strip()
        else:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
    out.append(cur)
    return out


def parse_srt(srt_path):
    # retorna lista de dicts: [{start, end, text}]
    pat_time = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\+?\d+)")
    segs, buf, t0, t1 = [], [], 0.0, 0.0
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        block = []
        for line in f:
            if line.strip() == "" and block:
                # parse block
                # expected: idx, time, text...
                if len(block) >= 2:
                    m = pat_time.search(block[1])
                    if m:
                        hh, mm, ss, ms = map(int, m.groups()[:4])
                        HH, MM, SS, MS = map(int, m.groups()[4:])
                        t0 = hh * 3600 + mm * 60 + ss + ms / 1000
                        t1 = HH * 3600 + MM * 60 + SS + MS / 1000
                        text = " ".join(x.strip() for x in block[2:] if x.strip())
                        segs.append({"start": t0, "end": t1, "text": text})
                block = []
            else:
                block.append(line.rstrip("\n"))
        # último bloque
        if block and len(block) >= 2:
            m = pat_time.search(block[1])
            if m:
                hh, mm, ss, ms = map(int, m.groups()[:4])
                HH, MM, SS, MS = map(int, m.groups()[4:])
                t0 = hh * 3600 + mm * 60 + ss + ms / 1000
                t1 = HH * 3600 + MM * 60 + SS + MS / 1000
                text = " ".join(x.strip() for x in block[2:] if x.strip())
                segs.append({"start": t0, "end": t1, "text": text})
    return segs


def _ts(t):
    # t en segundos -> "HH:MM:SS,mmm"
    ms = int(round((t - int(t)) * 1000))
    t = int(t)
    s = t % 60
    m = (t // 60) % 60
    h = t // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_words_from_segments(segs, out_srt, words_per_chunk=1, min_dur=0.12):
    """
    Convierte segmentos (start,end,text) en un SRT a nivel palabra.
    - words_per_chunk: cuántas palabras por cue (1 = una palabra)
    - min_dur: duración mínima por cue (seg); si no se llega, agrupa más palabras.
    """
    idx = 1
    lines = []
    for s in segs:
        t0, t1 = s["start"], s["end"]
        dur = max(0.01, t1 - t0)
        # tokenización simple (palabras con tildes) y conserva signos sueltos como "extras"
        raw = s["text"].strip()
        if not raw:
            continue
        # separa por espacios y quita comas/puntos del extremo
        tokens = []
        for tok in raw.split():
            tok_clean = tok.strip(".,;:!?¡¿()[]«»\"'")
            if tok_clean:
                tokens.append(tok_clean)

        if not tokens:
            continue

        # duración base por palabra
        base_dt = dur / len(tokens)

        # si base_dt < min_dur, agrupar automáticamente
        auto_group = max(1, math.ceil(min_dur / max(base_dt, 1e-6)))
        group = max(1, words_per_chunk, auto_group)

        # número de cues
        n_cues = math.ceil(len(tokens) / group)
        cue_dt = dur / n_cues

        for k in range(n_cues):
            start = t0 + k * cue_dt
            end = min(t1, start + cue_dt)
            chunk = tokens[k * group: (k + 1) * group]
            if not chunk:
                continue
            text = " ".join(chunk)

            lines.append(str(idx))
            lines.append(f"{_ts(start)} --> {_ts(end)}")
            lines.append(text)
            lines.append("")  # separador
            idx += 1

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_srt


def wrap_srt(in_srt, out_srt, max_chars=30):
    with open(in_srt, "r", encoding="utf-8", errors="ignore") as f:
        blocks, block = [], []
        for line in f:
            if line.strip() == "":
                if block: blocks.append(block); block = []
            else:
                block.append(line.rstrip("\n"))
        if block: blocks.append(block)

    pat_time = re.compile(r"\d+:\d+:\d+,\d+\s*-->\s*\d+:\d+:\d+,\d+")
    out = []
    for b in blocks:
        if len(b) >= 3 and pat_time.search(b[1]):
            idx = b[0]
            time = b[1]
            text = " ".join(x for x in b[2:] if x.strip())
            wrapped = textwrap.fill(text, width=max_chars)  # inserta saltos
            # limitar a 2 líneas (para shorts es lo ideal)
            lines = wrapped.splitlines()
            if len(lines) > 2:
                # junta el resto en la 2ª
                wrapped = "\n".join([lines[0], " ".join(lines[1:])])
            out += [idx, time, wrapped, ""]
        else:
            out += b + [""]

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    return out_srt


def srt_words_faster_whisper(audio_path, out_srt, model_size="medium", language="es",
                             device="cpu", compute_type="int8"):  # usa "cuda" si tienes GPU
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )

    def _ts(t):
        ms = int(round((t - int(t)) * 1000))
        t = int(t);
        s = t % 60;
        m = (t // 60) % 60;
        h = t // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    idx = 1
    lines = []
    MIN_DUR = 0.12  # asegura que no “parpadee”
    for seg in segments:
        words = getattr(seg, "words", None) or []
        if not words:
            # fallback: un cue por segmento
            lines += [str(idx), f"{_ts(seg.start)} --> {_ts(seg.end)}", seg.text.strip(), ""]
            idx += 1
            continue
        # fusiona palabras demasiado cortas
        buf = []
        for w in words:
            buf.append(w)
            # si el último bloque es muy corto, espera sumar otra palabra
            dur = (buf[-1].end - buf[0].start)
            if dur >= MIN_DUR:
                text = " ".join(x.word for x in buf).strip()
                lines += [str(idx), f"{_ts(buf[0].start)} --> {_ts(buf[-1].end)}", text, ""]
                idx += 1
                buf = []
        if buf:
            text = " ".join(x.word for x in buf).strip()
            lines += [str(idx), f"{_ts(buf[0].start)} --> {_ts(buf[-1].end)}", text, ""]
            idx += 1

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_srt


def burn_subs(input_mp4="tmp_with_music.mp4", srt="voz_words.srt", out="short_final.mp4"):
    out = str(Path(out).with_suffix(".mp4"))

    # estilo ASS
    style = f"Fontsize={FONT_SIZE},Outline={OUTLINE},Shadow={SHADOW},MarginV={MARGIN_V}"

    # Ruta absoluta en formato POSIX y ESCAPE de caracteres problemáticos para FFmpeg filtergraph
    srt_posix = Path(srt).resolve().as_posix()
    # En filtros de FFmpeg, ':' separa opciones, así que hay que escaparlo como '\:'
    # También escapamos comillas simples por seguridad.
    srt_escaped = srt_posix.replace(":", r"\:").replace("'", r"\'")

    # Importante: poner el filename entre comillas simples dentro del filtro
    vf = f"subtitles='{srt_escaped}':force_style='{style}'"

    cmd = f'''ffmpeg -y -i "{input_mp4}" -vf "{vf}" -c:a copy "{out}"'''
    run(cmd)
    return out

def burn_subs_with_music(
    input_mp4="tmp_base.mp4",
    srt="voz_wrapped.srt",   # o "voz_words.srt"
    music="music.mp3",
    out="short_final.mp4",
    music_vol=0.20,
    sc_threshold=0.03,
    sc_ratio=6,
    sc_attack=5,
    sc_release=250,
    a_bitrate="160k"
):
    style = f"Fontsize={FONT_SIZE},Outline={OUTLINE},Shadow={SHADOW},MarginV={MARGIN_V}"
    cmd = rf'''ffmpeg -y \
 -i "{input_mp4}" \
 -stream_loop -1 -i "{music}" \
 -filter_complex "[1:a]volume={music_vol}[bg];[0:a][bg]sidechaincompress=threshold={sc_threshold}:ratio={sc_ratio}:attack={sc_attack}:release={sc_release}:makeup=3:link=average[aout]" \
 -map 0:v -map "[aout]" -shortest \
 -vf "subtitles='{srt}':force_style='{style}'" \
 -c:v libx264 -c:a aac -b:a {a_bitrate} "{out}"'''
    run(cmd)
    return out




if __name__ == "__main__":
    audio = "voz.mp3"
    from datetime import datetime

    ts = time.strftime("%Y-%m-%d%H%M%S")
    fn_name = f"short-{ts}.mp4"
    # 1) SRT base para escenas (b-roll contextual por frase)
    srt_original = ensure_srt(audio, "voz.srt")
    segs_raw = parse_srt(srt_original)
    scene_segs = merge_short_segments(segs_raw, min_scene=2.0, max_scene=5.0)
    print(f"[i] escenas b-roll: {len(scene_segs)}")

    # 2) Generar SRT palabra-a-palabra real
    srt_words = ensure_words_srt(audio, "voz_words.srt")

    # 3) Construir video por escenas (b-roll coherente por frase)
    base = build_video_from_segments(scene_segs, audio)

    # 2) elegir SRT (por palabra o envuelto a 2 líneas)
    # srt_path = "voz_words.srt"
    srt_path = wrap_srt("voz.srt", "voz_wrapped.srt", max_chars=30)

    # 3) bajar música synthwave ALEATORIA
    music_path, meta = pick_and_download_openverse(out="music.mp3")
    print("[music]", meta)

    # si ya tenés tmp_base.mp4 (con tu voz):
    with_music = mix_music_into_video("tmp_base.mp4", music_path, out="tmp_with_music.mp4")

    # luego quemás subtítulos sobre ese archivo:
    final = burn_subs(with_music, srt_words, fn_name)
    print(f"[✔] Listo: {final}")
