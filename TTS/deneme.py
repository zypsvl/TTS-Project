import os
import torch
from TTS.api import TTS
import simpleaudio as sa
import sys

# TTS modelini başlat
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name)
def siiri_duz_metin_yap(siir):
    # Satırları böl ve aralarındaki gereksiz boşlukları temizle
    satirlar = siir.splitlines()
    # Satırları birleştir
    duz_metin = " ".join(satir.strip() for satir in satirlar if satir.strip())
    return duz_metin

def metni_parcalara_ayir(metin, max_harf=225):
    parcalar = []
    baslangic = 0
    metin_uzunlugu = len(metin)

    while baslangic < metin_uzunlugu:
        # 225 harflik bölgeyi al
        son = baslangic + max_harf

        # Eğer son, metin uzunluğunu aşarsa, sonu metnin uzunluğuna eşitle
        if son >= metin_uzunlugu:
            son = metin_uzunlugu
            parcalar.append(metin[baslangic:son])
            break

        # Eğer son karakter boşluk değilse, bir önceki boşluğu bul
        while son > baslangic and metin[son] != " ":
            son -= 1

        # Eğer boşluk bulunmazsa, 225. karakterde kes
        if son == baslangic:
            son = baslangic + max_harf

        # Parçayı ekle
        parcalar.append(metin[baslangic:son].strip())

        # Bir sonraki parça için başlangıcı ayarla
        baslangic = son + 1  # Boşluktan sonra başla

    return parcalar

def split_sentences(text):
    # Cümleleri ayır ve boşlukları temizle
    return [sentence.strip() for sentence in text.split('.') if sentence.strip()]

def run_tts(sentence, speaker_wav, language, output_path):
    try:
        # TTS sentezi yap
        wav_data = tts.tts(sentence, speaker_wav=speaker_wav, language=language)
        with open(output_path, "wb") as f:
            f.write(wav_data)
        print(f"Ses dosyası oluşturuldu: {output_path}")
    except Exception as e:
        print(f"Ses sentezi sırasında bir hata oluştu: {e}")

def play_audio(file_path):
    try:
        play_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = play_obj.play()
        play_obj.wait_done()  # Ses bitene kadar bekle
    except Exception as e:
        print(f"Ses oynatılırken bir hata oluştu: {e}")

def main():
    speaker_wav = r"C:\Users\zeyne\Desktop\bark\ATATÜRK\Atatürk1.wav"
    language = "tr"  # Türkçe için dil kodu

    if not os.path.isfile(speaker_wav):
        print(f"Hata: '{speaker_wav}' dosyası bulunamadı.")
        return

    while True:
        text = input("Metni girin (çıkmak için 'q' ya basın): ")
        if text.lower() == 'q':
            break

        # Şiiri düz metin haline getir
        duz_metin = siiri_duz_metin_yap(text)

        # Metni parçalara ayıralım
        parcalar = metni_parcalara_ayir(duz_metin)

        for i, parca in enumerate(parcalar):
            output_path = f"temp_output_sentence_{i}.wav"
            run_tts(parca, speaker_wav, language, output_path)
            if os.path.isfile(output_path):  # Dosyanın oluşturulduğunu kontrol et
                play_audio(output_path)
                os.remove(output_path)
            else:
                print(f"Hata: '{output_path}' dosyası oluşturulamadı.")

if __name__ == "__main__":
    main()
