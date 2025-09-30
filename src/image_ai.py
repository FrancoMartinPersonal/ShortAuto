import requests, time, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def build_image_prompt_from_sentence(sentence: str):
    """
    Generador de prompt genérico (tema-agnóstico) para imagen IA.
    - Usa la frase completa (contexto).
    - Estilo: ilustración/infografía limpia (evita realismo fotográfico).
    - Negative prompt fuerte para bloquear gore/NSFW/texto/logos, etc.
    """
    # 1) limpieza mínima
    s = " ".join(sentence.strip().split())
    s = s.replace("¿", "").replace("?", "").replace("¡", "").replace("!", "")

    # 2) prompt neutro (en inglés para mejor respuesta de SDXL),
    #    incluyendo la frase original entre comillas (cualquier idioma).
    base_style = (
        "clean conceptual illustration, minimal infographic, isometric elements, "
        "soft studio lighting, neutral background, high detail, 9:16 aspect ratio, "
        "no caption text, no watermark"
    )
    prompt = (
        f"{base_style}. Depict the concept described here (do not write any words): \"{s}\""
    )

    # 3) negative prompt genérico y amplio (tema-agnóstico)
    negative = (
        "blood, gore, violence, injury, surgery, organs, realistic human flesh, "
        "nsfw, nudity, offensive, disturbing, scary, horror, "
        "photographic realism, photo, render artifacts, watermark, logo, signature, "
        "text, captions, subtitles, lowres, blurry, deformed, distorted, extra limbs"
    )

    # 4) (opcional) acotar longitud si el texto es larguísimo
    if len(prompt) > 600:
        prompt = prompt[:600]

    print("[ia-prompt] +", prompt)
    print("[ia-neg]   -", negative)
    return prompt, negative


# Reemplaza tu generate_image_hf por este
def generate_image_hf(
    sentence_text,
    out_path="tmp_broll/ia_first.jpg",
    models=None,
    retries_per_model=2,
    wait_s=6,
):
    if not HF_TOKEN:
        raise RuntimeError("Falta HF_TOKEN en el .env")

    models = models or [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/sd-turbo",
    ]

    pos_prompt, neg_prompt = build_image_prompt_from_sentence(sentence_text)

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": pos_prompt,
        "parameters": {
            "negative_prompt": neg_prompt,
            "num_inference_steps": 28,
            "guidance_scale": 7.0,
        },
    }

    last_err = None
    for model in models:
        url = f"https://api-inference.huggingface.co/models/{model}"
        print(f"[hf] intentando modelo: {model}")
        for attempt in range(1, retries_per_model + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                if r.status_code == 503:
                    et = (r.json().get("estimated_time", wait_s) if r.headers.get("content-type","").startswith("application/json") else wait_s)
                    print(f"[hf] 503 warming-up ({model}), retry en {int(et)}s…")
                    time.sleep(min(max(3, int(et)), 20))
                    continue
                if r.status_code in (401,403,404):
                    print(f"[hf] {r.status_code} en {model}: {r.text[:200]}")
                    break
                r.raise_for_status()
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(r.content)
                print(f"[hf] imagen generada con {model} -> {out_path}")
                return out_path
            except Exception as e:
                last_err = e
                print(f"[hf] error ({model}, intento {attempt}): {e}")
                time.sleep(wait_s)
    raise last_err or RuntimeError("No se pudo generar imagen IA con ninguno de los modelos")