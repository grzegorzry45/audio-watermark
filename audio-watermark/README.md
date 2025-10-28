# Audio Watermark — MVP (Render-ready)

Prosta aplikacja webowa:
- **/ (UI)**: wgraj WAV/MP3, dodaj watermark (tekst), wybierz WAV lub MP3 256 kbps.
- **/embed (POST)**: osadzanie watermarku (niesłyszalny, STFT, 1–4 kHz).
- **/detect (POST)**: detekcja (wymaga podania tego samego tekstu).

## Deploy na Render
1. Zrób repo na GitHub i wrzuć pliki z tego projektu.
2. Na https://render.com → **New +** → **Web Service** → wybierz to repo.
3. Render wykryje `Dockerfile`, zbuduje obraz (Python + ffmpeg) i uruchomi usługę.
4. Otwórz publiczny URL i testuj.

## Lokalnie
```bash
pip install -r requirements.txt
export PORT=10000
uvicorn app:app --host 0.0.0.0 --port $PORT
```
Wejdź na http://localhost:10000/

## Uwaga
To **MVP**. Najlepiej działa, gdy wynik eksportujesz do **WAV** — ma największą szansę przetrwać transkodowanie (YouTube/Spotify). MP3 256 kbps działa, ale watermark może być minimalnie słabszy. Jeśli chcesz większej odporności, zwiększ `base_alpha` w `embed_watermark` (kosztem ryzyka słyszalności).
