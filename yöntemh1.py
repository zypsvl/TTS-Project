import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio
import os

def metni_parcalara_ayir(metin, max_harf=225):
    parcalar = []
    baslangic = 0
    metin_uzunlugu = len(metin)
    noktalama_isaretleri = {'.', ',', ';', '!', '?'}

    while baslangic < metin_uzunlugu:
        son = baslangic + max_harf

        if son >= metin_uzunlugu:
            son = metin_uzunlugu
            parcalar.append(metin[baslangic:son].strip())
            break

        while son > baslangic and metin[son] not in noktalama_isaretleri and metin[son] not in [' ']:
            son -= 1

        if son == baslangic:
            son = baslangic + max_harf
        else:
            son += 1

        parcalar.append(metin[baslangic:son].strip())
        baslangic = son

    return parcalar

def siiri_duz_metin_yap(siir):
    satirlar = siir.splitlines()
    duz_metin = ""

    for i, satir in enumerate(satirlar):
        satir = satir.strip()
        if not satir:
            continue

        if i < len(satirlar) - 1 and satir[-1] not in ['.', ',']:
            satir += ','

        duz_metin += satir + " "

    return duz_metin.strip()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = XttsConfig()
    config.load_json(xtts_config)

    model = Xtts.init_from_config(config)
    print("XTTS modeli yükleniyor...")
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)

    if torch.cuda.is_available():
        model.cuda()

    print("Model başarıyla yüklendi!")
    return model

def run_tts(model, tts_text, speaker_audio_file):
    try:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs
        )

        out = model.inference(
            text=tts_text,
            language='tr',
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=model.config.temperature,
            length_penalty=model.config.length_penalty,
            repetition_penalty=model.config.repetition_penalty,
            top_k=model.config.top_k,
            top_p=model.config.top_p
        )

        audio_tensor = torch.tensor(out['wav']).unsqueeze(0)
        audio_tensor = normalize_audio(audio_tensor)
        return audio_tensor
    except Exception as e:
        print(f"TTS işleminde hata oluştu: {e}")
        return None

def normalize_audio(audio_tensor):
    audio_max = torch.max(audio_tensor)
    audio_min = torch.min(audio_tensor)
    if audio_max != audio_min:
        audio_tensor = (audio_tensor - audio_min) / (audio_max - audio_min)
    return audio_tensor

def save_wav_file(wav_tensor, output_path):
    try:
        # Eğer ses tensor'ü 1D ise, 2D'ye dönüştür
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        torchaudio.save(output_path, wav_tensor, 24000)
        print(f"Ses dosyası başarıyla kaydedildi: {output_path}")
    except Exception as e:
        print(f"Ses dosyası kaydedilirken hata oluştu: {e}")

def main():
    i = 1

    xtts_checkpoint = r'C:\Users\zeyne\Desktop\bark\AtatürkEğitilmişModel3\best_model.pth'
    xtts_config = r'C:\Users\zeyne\Desktop\bark\AtatürkEğitilmişModel3\config (1).json'
    xtts_vocab = r'C:\Users\zeyne\Desktop\bark\AtatürkEğitilmişModel3\vocab (1).json'
    speaker_audio_file = r'C:\Users\zeyne\Desktop\bark\ATATÜRK\0902 (1).MP3'

    model = load_model(xtts_checkpoint, xtts_config, xtts_vocab)

    while True:
        print("Lütfen bir metin giriniz (Ctrl+D ile sonlandırabilirsiniz):")

        metinler = []
        while True:
            try:
                satir = input()
                if not satir.strip():
                    break
                metinler.append(satir)
            except EOFError:
                if metinler:
                    break
                else:
                    print("Boş metin girildi. Çıkılıyor...")
                    return

        text = "\n".join(metinler).strip()

        if not text:
            print("Boş metin girildi. Çıkılıyor...")
            break

        duz_metin = siiri_duz_metin_yap(text)
        parcalar = metni_parcalara_ayir(duz_metin)

        combined_wav = []
        try:
            for parca in parcalar:
                print(f"İşleniyor: {parca}...")
                wav_tensor = run_tts(model, parca, speaker_audio_file)
                if wav_tensor is not None:
                    combined_wav.append(wav_tensor)
                else:
                    print("Ses dosyası oluşturulamadı.")
        except Exception as e:
            print(f"Döngü başarısız: {e}")

        if not combined_wav:
            print("Birleştirilecek ses dosyası bulunamadı.")
            continue

        combined_wav_tensor = torch.cat(combined_wav, dim=1)

        output_path = f'C:\\Users\\zeyne\\Desktop\\bark\\output_audio\\genclige_hitabe{i}.wav'

        save_wav_file(combined_wav_tensor, output_path)
        i += 1

if __name__ == "__main__":
    main()
