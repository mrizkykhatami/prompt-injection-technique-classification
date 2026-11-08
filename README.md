# Klasifikasi Multikelas Teknik Serangan Prompt Injection

Kode, data turunan, dan hasil eksperimen untuk paper **"Klasifikasi Multikelas Teknik Serangan *Prompt Injection*: Perbandingan Pengklasifikasi Klasik pada Representasi Leksikal dan Semantik"** (Teknik Informatika, Universitas Padjadjaran).

Penelitian ini memetakan kategori serangan pada [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) (konfigurasi `core`) ke enam kelas tingkat teknik, lalu membandingkan empat algoritma pada tiga representasi teks (12 konfigurasi).

| Kode | Kelas |
| --- | --- |
| K0 | Benign |
| K1 | Instruction Override |
| K2 | Role-play and Conversational Jailbreak |
| K3 | Obfuscation and Token Smuggling |
| K4 | Prompt and Information Extraction |
| K5 | Code Execution, Indirect and Agent Injection |

- **Algoritma:** Naive Bayes, SVM, Random Forest, MLP
- **Representasi:** R1 TF-IDF kata (1–2 gram), R2 TF-IDF karakter (char_wb 3–5 gram), R3 *sentence embedding* `all-MiniLM-L12-v2`
- **Protokol:** hiperparameter dipilih pada data validasi, model dilatih ulang pada latih+validasi, data uji dievaluasi satu kali. RF dan MLP dijalankan dengan lima *seed* (42–46).
- **Statistik:** macro-F1 dengan CI bootstrap (B = 1.000), bootstrap berpasangan, McNemar eksak dengan koreksi Holm.
- **Audit tambahan:** per sumber data, sumber yang disisihkan (*held-out*), NotInject, *hard negative*, varian biner, derau label.

## Isi repositori

```
notebooks/
  Notebook_Final_rev3.ipynb   notebook yang menghasilkan semua angka di paper (dengan output)
  Notebook_Final_rev4.ipynb   rev3 + sel 15B yang membuat Gambar 2–4 langsung dari hasil run
mapping.csv                   pemetaan 29 kategori asli ke kelas K0–K5
inputs/
  label_audit_wildguard.csv   penilaian tim atas 13 positif palsu wildguard_judgecomp
results_rev3/                 tabel CSV dan manifest dari run rev3 (sumber resmi angka paper)
paper_figures/                Gambar 1–4 paper (PNG 300 dpi) dan sumber draw.io Gambar 1
scripts/make_paper_figures.py skrip pembuat gambar paper
requirements.txt              versi pustaka yang dipakai
```

## Menjalankan ulang

1. Salin `mapping.csv` dan folder `inputs/` ke `MyDrive/prompt_injection_project/` di Google Drive.
2. Buka notebook di Google Colab, lalu pilih **Runtime → Restart session and run all**. Sel pertama memasang versi pustaka yang sama dengan `requirements.txt`.
3. Hasil tersimpan di `prompt_injection_project/results_rev4/` (rev4) atau `results/` (rev3).

Run rev3 memakai runtime TPU v5e-1 Colab. Komputasi berjalan di CPU host (2 vCPU) karena scikit-learn dan sentence-transformers tidak memakai TPU. Semua metrik model bersifat deterministik dengan *seed* dan versi pustaka yang sama; hanya waktu inferensi (kolom `ms/1.000`) yang bergantung pada perangkat keras.

## Data

Dataset asli tidak disertakan; notebook mengunduhnya langsung dari Hugging Face. Lisensi dataset mengikuti ketentuan pada *dataset card*-nya. Benchmark NotInject diunduh dari [leolee99/NotInject](https://huggingface.co/datasets/leolee99/NotInject).

## Lisensi

Kode dirilis di bawah lisensi MIT (lihat `LICENSE`). Dataset pihak ketiga mengikuti lisensinya masing-masing.

## Tim

- Muhammad Rizky Khatami
- Hamzah Abdillah Gabriela
- Renadi Wilantara
