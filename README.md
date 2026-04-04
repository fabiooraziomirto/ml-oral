# ML-Bitto

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PLACEHOLDER/ML-Bitto/blob/main/reproduce.ipynb)

Pipeline PyTorch per classificazione binaria di immagini cliniche/orali.

Il repository gestisce:
- caricamento di dataset pubblici e clinici
- split train/val/test con patient-level split per il dataset clinico
- training del modello
- valutazione con ROC, confusion matrix, calibration curve e Brier score
- Grad-CAM opzionale
- cross-validation opzionale

## Reproducing results

Comando rapido per riprodurre la pipeline end-to-end:

```bash
bash reproduce.sh
```

Flag opzionali disponibili:
- `--skip-tests`
- `--skip-install`

Note:
- la run in debug usa dati sintetici e non richiede dataset reali
- l'output atteso è una sequenza di step verdi con metriche finali stampate a console
- tempo stimato su CPU per la modalità debug: circa 2-3 minuti
- `reproduce.sh` e `Makefile` sono pensati per Linux e macOS; Windows è fuori scope
- il notebook [reproduce.ipynb](reproduce.ipynb) può essere eseguito su Colab senza setup locale
- prima della pubblicazione, il link `PLACEHOLDER` del badge Colab e del notebook va sostituito con l'URL GitHub reale del repository

## Requisiti

Python 3.10.12 usato e testato.

Altre versioni di Python potrebbero funzionare, ma al momento non sono state
validate in questo repository.

Installazione dipendenze:

```bash
pip install -r requirements.txt
```

## Come lanciare

Esecuzione rapida in debug, senza dataset reali:

```bash
python main.py --debug
```

Training su dataset pubblico soltanto:

```bash
python main.py --mode public_only --public data/public_dataset_1
```

Training con dataset clinico:

```bash
python main.py --clinical data/clinical_dataset
```

Opzioni utili:

```bash
python main.py --epochs 30 --lr 5e-4 --batch-size 16 --monitor recall
python main.py --clinical data/clinical_dataset --gradcam
python main.py --clinical data/clinical_dataset --cross-val
```

Flag principali:
- `--debug`: usa dati sintetici
- `--mode public_only|clinical`: forza la modalità
- `--public PATH`: path al dataset pubblico
- `--clinical PATH`: path al dataset clinico
- `--gradcam`: genera visualizzazioni Grad-CAM
- `--cross-val`: esegue repeated holdout cross-validation
- `--monitor accuracy|recall|f1|auc`: metrica di early stopping

## Struttura dei dataset

### Dataset pubblico

Struttura attesa:

```text
data/public_dataset_1/
├── images/
└── labels.csv
```

`labels.csv` deve contenere almeno queste colonne:
- `image_name`
- `label`

Esempio:

```csv
image_name,label
img_001.jpg,0
img_002.jpg,1
```

### Dataset clinico

Struttura attesa:

```text
data/clinical_dataset/
├── images/
└── metadata.csv
```

`metadata.csv` deve contenere almeno:
- `image_name`
- `biopsy_diagnosis`

Colonne opzionali supportate:
- `id`
- `clinician_diagnosis`
- `lesion_type`
- `location`
- `patient_id`

Esempio:

```csv
id,patient_id,image_name,biopsy_diagnosis,clinician_diagnosis,lesion_type,location
pt001,p001,img_001.jpg,1,0,oral_cancer,buccal
pt002,p002,img_002.jpg,0,0,benign,palate
```

`patient_id` non è obbligatorio, ma è fortemente consigliato perché permette lo split per paziente ed evita leakage tra train/val/test.

## Come scaricare i dataset

Il repository non include uno script di download automatico.

Quindi il flusso è:
1. scarica i dataset dalle fonti che stai usando
2. crea le cartelle `data/public_dataset_1` e/o `data/clinical_dataset`
3. copia le immagini dentro `images/`
4. crea `labels.csv` o `metadata.csv` con le colonne richieste sopra

Se vuoi, nel prossimo passaggio posso aggiungere anche una sezione con un esempio pratico di preparazione dataset a partire da uno zip o da un CSV grezzo.