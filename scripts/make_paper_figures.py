# Figur versi Indonesia: label diterjemahkan, desimal koma. Nilai sama dengan versi Inggris.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","DejaVu Serif"],"font.size":7})
c = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")
cbfmt = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
O = "paper_figures/"

# ---------- Gambar 1: alur kerja
fig, ax = plt.subplots(figsize=(3.45, 4.1), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 12.2); ax.axis("off")
steps = [
 ("neuralchemy Prompt-injection-dataset (konfigurasi core)", "pembagian resmi 4.391 / 941 / 942; masukan = kolom teks saja"),
 ("Pemetaan kategori ke kelas", "31 kategori → 6 kelas teknik K0–K5;\nkeluarkan control dan edge_case (59 / 16 / 14 baris)"),
 ("Representasi teks (tanpa melibatkan data uji)", "TF-IDF di-fit pada data latih, lalu latih+validasi (model akhir)\nkata 1–2 gram | karakter 3–5 gram | all-MiniLM-L12-v2"),
 ("Pengklasifikasi", "Naive Bayes | SVM | Random Forest | MLP\n(4 × 3 = 12 konfigurasi)"),
 ("Pencarian hiperparameter pada data validasi", "kriteria pemilihan: macro-F1"),
 ("Latih ulang pada data latih + validasi; satu kali evaluasi uji", "RF dan MLP: 5 seed (42–46)"),
 ("Evaluasi dan analisis", "macro-F1, CI bootstrap, bootstrap berpasangan, McNemar + Holm;\nper kelas, kategori, sumber; sumber disisihkan; NotInject; biner"),
]
h = 1.36; gap = 0.33; y = 12.0
for i,(t,s) in enumerate(steps):
    y0 = y - h
    ax.add_patch(FancyBboxPatch((0.25, y0), 9.5, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc="#EEF3FA" if i % 2 == 0 else "#F7F7F7", ec="#2F4B7C", lw=0.7))
    ax.text(5, y0 + h - 0.36, t, ha="center", va="center", fontsize=7.0, weight="bold", color="#1B2A41")
    ax.text(5, y0 + 0.47, s, ha="center", va="center", fontsize=6.1, color="#222222", linespacing=1.15)
    if i < len(steps) - 1:
        ax.annotate("", xy=(5, y0 - gap + 0.03), xytext=(5, y0 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#2F4B7C", mutation_scale=7))
    y = y0 - gap
plt.savefig(O + "fig1_workflow.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Gambar 2: confusion matrix ternormalisasi SVM / R2 (nilai dari figur notebook, 2 desimal)
CM = np.array([[0.93,0.01,0.06,0.00,0.00,0.00],
               [0.00,0.97,0.02,0.00,0.00,0.00],
               [0.10,0.02,0.86,0.02,0.00,0.00],
               [0.00,0.00,0.00,0.97,0.00,0.03],
               [0.00,0.07,0.00,0.00,0.93,0.00],
               [0.00,0.01,0.01,0.01,0.01,0.96]])
K = ["K0","K1","K2","K3","K4","K5"]
fig, ax = plt.subplots(figsize=(3.45, 2.85), dpi=300)
im = ax.imshow(CM, cmap="Blues", vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, c(CM[i,j]), ha="center", va="center", fontsize=6.3, color="white" if CM[i,j] > 0.5 else "#222222")
ax.set_xticks(range(6)); ax.set_xticklabels(K); ax.set_yticks(range(6)); ax.set_yticklabels(K)
ax.set_xlabel("Kelas prediksi"); ax.set_ylabel("Kelas sebenarnya"); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format=cbfmt); cb.ax.tick_params(labelsize=6)
plt.savefig(O + "confusion.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Gambar 3: recall per kategori asli (dari original_category_level_recall.csv, seed 42)
cfg = ["NB R1","NB R2","NB R3","SVM R1","SVM R2","SVM R3","RF R1","RF R2","RF R3","MLP R1","MLP R2","MLP R3"]
e = 1/8
R = [
 ("K0","benign",381,               [.91,.93,.89,.93,.93,.94,.93,.92,.96,.93,.95,.95]),
 ("K1","context_confusion",6,      [5/6,4/6,0,5/6,4/6,4/6,1,5/6,1,5/6,4/6,4/6]),
 ("K1","direct_injection",314,     [.86,.99,.85,.99,.98,.97,.99,.98,.98,1.0,.98,.97]),
 ("K1","instruction_override",4,   [1,1,0,1,1,.25,1,1,.25,1,1,.5]),
 ("K1","output_manipulation",5,    [.8,1,.2,1,1,1,1,1,1,1,1,1]),
 ("K1","payload_injection",2,      [1,0,0,1,1,1,1,1,1,1,1,1]),
 ("K1","prompt_injection",4,       [.75,1,0,1,1,1,1,.75,.75,1,1,.75]),
 ("K1","response_manipulation",4,  [1,1,0,1,.75,1,1,1,.75,1,.75,.75]),
 ("K1","system_manipulation",4,    [.75,1,0,1,1,1,1,1,1,1,1,1]),
 ("K2","jailbreak",50,             [.68,.62,.78,.76,.86,.68,.74,.86,.38,.76,.72,.50]),
 ("K2","many_shot",1,              [1,0,1,1,1,1,0,1,1,1,0,1]),
 ("K2","multi_turn",4,             [1,.75,1,1,1,1,1,1,1,1,1,1]),
 ("K2","persona_replacement",3,    [2/3,1/3,2/3,2/3,2/3,2/3,2/3,2/3,2/3,2/3,2/3,2/3]),
 ("K3","encoding",30,              [.8,25/30,.9,26/30,29/30,29/30,25/30,29/30,23/30,25/30,.9,.9]),
 ("K3","encoding_obfuscation",1,   [0,0,0,1,1,1,1,0,0,0,1,1]),
 ("K3","token_injection",1,        [1,1,1,1,1,1,0,1,0,1,1,1]),
 ("K3","token_smuggling",7,        [3/7,4/7,1/7,5/7,1,5/7,3/7,1,2/7,4/7,1,4/7]),
 ("K4","prompt_extraction",3,      [1/3,1/3,2/3,1,1,2/3,2/3,1,0,1,1,2/3]),
 ("K4","system_extraction",2,      [0,0,1,0,.5,.5,0,0,0,0,.5,0]),
 ("K4","training_extraction",10,   [.9,.5,.5,.9,1,.8,.8,1,.4,.9,.9,.8]),
 ("K5","adversarial",79,           [76/79,74/79,61/79,74/79,76/79,75/79,72/79,74/79,69/79,73/79,76/79,73/79]),
 ("K5","agent_manipulation",8,     [5*e,5*e,0,7*e,1,7*e,5*e,6*e,3*e,5*e,1,5*e]),
 ("K5","indirect_injection",1,     [0,0,0,1,0,0,0,0,0,0,0,0]),
 ("K5","rag_poisoning",4,          [.5,.75,0,.75,1,.75,.75,.75,.5,.75,1,1]),
]
M = np.array([r[3] for r in R])
fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=300)
im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, c(M[i,j]), ha="center", va="center", fontsize=5.4, color="white" if M[i,j] > 0.6 else "#222222")
ax.set_yticks(range(len(R))); ax.set_yticklabels([f"{k} · {cat} (n={n})" for k,cat,n,_ in R], fontsize=5.8)
ax.set_xticks(range(12)); ax.set_xticklabels(cfg, rotation=90, fontsize=5.8)
ax.set_xlabel("Algoritma / Representasi", fontsize=6.5); ax.set_ylabel("Kelas hasil pemetaan · kategori asli", fontsize=6.5)
ax.tick_params(length=0)
for k in (2.5, 5.5, 8.5): ax.axvline(k, color="white", lw=1.0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, format=cbfmt); cb.ax.tick_params(labelsize=5.5); cb.set_label("Recall", fontsize=6)
plt.savefig(O + "recall_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Gambar 4: akurasi per sumber data
sources = ["neuralchemy_v1","hackaprompt","harmbench","harmbench_benign","wildguard_judgecomp","original"]
n = {"neuralchemy_v1":"518","hackaprompt":"310","harmbench":"16","harmbench_benign":"22","wildguard_judgecomp":"43","original":"19"}
acc = {
 "NB R1":  [0.9421,0.8613,0.5625,0.2727,0.5349,0.5263], "NB R2":  [0.9305,0.9968,0.3750,0.6364,0.5814,0.4211],
 "NB R3":  [0.8359,0.8613,0.8125,0.5455,0.3953,0.5263], "SVM R1": [0.9653,0.9935,0.6875,0.4091,0.6512,0.6316],
 "SVM R2": [0.9826,0.9903,0.8125,0.4091,0.6512,0.6316], "SVM R3": [0.9595,0.9839,0.5000,0.5455,0.6279,0.5789],
 "RF R1":  [0.9479,0.9968,0.5000,0.4545,0.6744,0.5263], "RF R2":  [0.9768,0.9839,0.7500,0.4545,0.6047,0.5789],
 "RF R3":  [0.8958,0.9871,0.0625,0.9091,0.6047,0.4737], "MLP R1": [0.9556,0.9968,0.6250,0.4091,0.6512,0.5789],
 "MLP R2": [0.9788,0.9871,0.4375,0.5455,0.7209,0.5789], "MLP R3": [0.9382,0.9774,0.5000,0.7727,0.5349,0.4211],
}
rows = list(acc); M = np.array([acc[r] for r in rows])
fig, ax = plt.subplots(figsize=(3.45, 3.3), dpi=300)
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, c(M[i,j]), ha="center", va="center", fontsize=5.6, color="white" if M[i,j] > 0.6 else "#111111")
labels = ["neuralchemy\n_v1", "hackaprompt", "harmbench", "harmbench\n_benign", "wildguard\n_judgecomp", "original"]
ax.set_xticks(range(6)); ax.set_xticklabels([f"{l}\n(n={n[s]})" for l,s in zip(labels,sources)], fontsize=5.4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=6)
for k in (2.5, 5.5, 8.5): ax.axhline(k, color="white", lw=1.2)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=cbfmt); cb.ax.tick_params(labelsize=5.5); cb.set_label("Akurasi enam kelas", fontsize=6)
ax.set_xlabel("Sumber data (data uji)", fontsize=6.5)
plt.savefig(O + "fig4_source_accuracy.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
print("ok")
