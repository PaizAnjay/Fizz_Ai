import os
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from groq import Groq
import edge_tts

app = FastAPI()

# Mengambil API key dari pengaturan rahasia Hugging Face
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/chat")
async def process_audio(audio: UploadFile = File(...)):
    # 1. Simpan audio dari ESP32 ke file sementara (WAV/PCM)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    try:
        # 2. STT (Telinga): Ubah Suara jadi Teks pakai Groq Whisper
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                language="id"
            )
        user_text = transcription.text
        print(f"User: {user_text}")

        # 3. LLM (Otak): Dapatkan Jawaban AI pakai Groq Llama-3.1
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI di dalam perangkat smart speaker bernama Espy. Jawab dengan sangat singkat, hangat, dan gunakan bahasa Indonesia sehari-hari. DILARANG menggunakan format markdown (seperti bintang atau pagar) karena jawabanmu akan dibaca menjadi suara."},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.1-8b-instant",
        )
        ai_response = chat_completion.choices[0].message.content
        print(f"AI: {ai_response}")

        # 4. TTS (Mulut): Ubah Teks jadi Suara MP3 pakai Edge-TTS
        output_audio_path = tempfile.mktemp(suffix=".mp3")
        # id-ID-ArdiNeural untuk pria, id-ID-GadisNeural untuk wanita
        tts = edge_tts.Communicate(ai_response, "id-ID-ArdiNeural") 
        await tts.save(output_audio_path)

        # 5. Kirim audio MP3 kembali ke ESP32
        return FileResponse(output_audio_path, media_type="audio/mpeg", filename="jawaban.mp3")

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Bersihkan memori server
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
          
