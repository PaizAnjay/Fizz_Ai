import os
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from groq import Groq
import edge_tts

app = FastAPI()

# Mengambil API key dari Environment Vercel
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/")
def read_root():
    return {"status": "Server ESP32 AI Aktif!"}

@app.post("/chat")
async def process_audio(audio: UploadFile = File(...)):
    # Vercel hanya mengizinkan kita menulis file di folder /tmp
    temp_audio_path = f"/tmp/{audio.filename}"
    output_audio_path = "/tmp/jawaban.mp3"

    # 1. Simpan audio dari ESP32
    with open(temp_audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        # 2. STT: Ubah Suara jadi Teks pakai Groq Whisper
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                language="id"
            )
        user_text = transcription.text
        print(f"User: {user_text}")

        # 3. LLM: Dapatkan Jawaban AI pakai Groq Llama
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI. Jawab dengan sangat singkat, maksimal 2 kalimat, dan gunakan bahasa Indonesia sehari-hari."},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.1-8b-instant",
        )
        ai_response = chat_completion.choices[0].message.content
        print(f"AI: {ai_response}")

        # 4. TTS: Ubah Teks jadi Suara MP3
        tts = edge_tts.Communicate(ai_response, "id-ID-ArdiNeural") 
        await tts.save(output_audio_path)

        # 5. Kirim kembali ke ESP32
        return FileResponse(output_audio_path, media_type="audio/mpeg", filename="jawaban.mp3")

    except Exception as e:
        return {"error": str(e)}
