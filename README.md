# WeatherAUS — Data Science Project

Progetto di Data Science sul dataset **Rain in Australia (WeatherAUS)**. Un unico dataset meteorologico viene affrontato con tre famiglie di tecniche: **classificazione**, **clustering** e **analisi di serie temporali**. In ogni analisi il protocollo sperimentale è pensato per evitare risultati ottimistici (leakage temporale, selezione sul test, scenari non realizzabili).

Progetto per il corso di Data Science, Laurea Magistrale in Ingegneria Informatica e dell'Automazione, Università Politecnica delle Marche (UNIVPM), A.A. 2025/2026.

📄 **Relazione completa:** [Relazione_WeatherAUS.pdf](Relazione_WeatherAUS.pdf), con la descrizione dettagliata di metodi, protocollo sperimentale e risultati.

---

## Dataset

I dati provengono dal dataset pubblico **[Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)** (Kaggle), basato su osservazioni del Bureau of Meteorology australiano.

| | |
|---|---|
| Osservazioni | 145 460 giornaliere (142 193 con target valido) |
| Stazioni | 49 località |
| Periodo | 1 novembre 2007 – 25 giugno 2017 |
| Variabili | 23 (temperature, umidità, pressione, vento, pioggia, nuvolosità…) |
| Target (classificazione) | `RainTomorrow`, pioggia il giorno successivo (22,4% di casi positivi, rapporto circa 3,5:1) |

---

## Struttura del repository

```
WeatherAUS_DataScience_Project/
├── classification_and_clustering/
│   ├── eda_data_analysis.ipynb                    # Analisi esplorativa (EDA)
│   ├── A_baseline_classification.ipynb              # Classificazione, configurazione A
│   ├── B_base_norid_classification.ipynb            # Classificazione, configurazione B
│   ├── C_rec_var_corr_classification.ipynb          # Classificazione, configurazione C
│   ├── D_reparam_collinearity_classification.ipynb  # Classificazione, configurazione D (modello finale)
│   ├── ablation.json                              # Risultati delle 4 configurazioni (generato dai notebook)
│   ├── MetaCost.py                                # Implementazione dell'algoritmo MetaCost
│   └── clustering.ipynb                           # Clustering dei regimi meteorologici
├── temporal_series/
│   └── time_series.ipynb                          # Serie temporali (SARIMA) su Temp3pm a Sydney
├── dataset/                                       # weatherAUS.csv (da scaricare, non versionato)
├── Relazione_WeatherAUS.pdf                       # Relazione del progetto
├── README.md
└── .gitignore
```

`MetaCost.py` deve trovarsi nella stessa cartella dei notebook di classificazione, che lo importano.

---

## Le analisi

### 1. Analisi esplorativa (EDA)

`eda_data_analysis.ipynb` esamina la struttura del dataset e individua le scelte da compiere prima della modellazione.

- **Valori mancanti.** Eliminate le 3 267 righe senza target. `Sunshine`, `Evaporation`, `Cloud9am` e `Cloud3pm` mancano per intere stazioni (mancanza strutturale).
- **Struttura temporale.** Il dataset è un panel di 49 serie storiche. La pioggia è autocorrelata e `RainTomorrow` di un giorno coincide con `RainToday` del giorno successivo. Per questo nella classificazione lo split deve essere cronologico.
- **Sbilanciamento** del target, circa 3,5:1. L'accuracy è quindi una metrica fuorviante.
- **Multicollinearità** moderata fra temperature e pressioni (Condition Index 21,2, cinque VIF > 10). Una riparametrizzazione in livelli e variazioni (`TempRange`, `PressureTrend`, `HumidityTrend`, `WindTrend`) porta il Condition Index a 4,3.
- **Ridondanza** fra `Rainfall` e `RainToday`, risolta con `RainfallLog = log(1 + Rainfall)`.

### 2. Classificazione binaria: *pioverà domani?*

Quattro notebook con **la stessa pipeline**, che differiscono solo per l'insieme di feature (**studio di ablazione**: ogni configurazione introduce una sola modifica rispetto alla precedente).

| Config. | Notebook | Modifica | Feature |
|---|---|---|---|
| A | `baseline_classification.ipynb` | Scartate le 4 colonne sparse | 135 |
| B | `base_norid_classification.ipynb` | `Rainfall` + `RainToday` sostituite da `RainfallLog` | 133 |
| C | `rec_var_corr_classification.ipynb` | Recuperate `Sunshine` e `Cloud3pm` con indicatore di mancanza | 137 |
| D | `reparam_collinearity_classification.ipynb` | Riparametrizzazione anti-collinearità | 133 |

**Protocollo**

- **Split cronologico** 80/20 su una data di taglio: training fino al 9 novembre 2015 (113 748 righe), test dal 10 novembre 2015 (28 445 righe).
- **Validazione incrociata temporale** con `TimeSeriesSplit` a 5 fold ed embargo di 500 righe fra addestramento e validazione.
- Preprocessing (imputazione, standardizzazione, one-hot) dentro una `Pipeline`, riaddestrato in ogni fold.
- **Selezione per PR-AUC**, una metrica che non dipende dalla soglia ed è adatta a classi sbilanciate.
- **Matrice dei costi** con un falso negativo che costa 3 volte un falso allarme (soglia teorica 0,25). La soglia di decisione è la mediana delle soglie a costo minimo sui 5 fold.
- Confronto con due baseline: **maggioranza** e **persistenza** (domani piove se oggi è piovuto).
- Il test set è usato una sola volta, per la valutazione finale.

**Risultati**

- **XGBoost** è il modello migliore in tutte e quattro le configurazioni (seguono Random Forest, regressione logistica e SVM lineare).
- **Configurazione D (modello finale).** PR-AUC in CV 0,741 (0,744 dopo l'ottimizzazione degli iperparametri). Sul test, con soglia 0,20: **ROC-AUC 0,886**, **PR-AUC 0,737**, **Recall 0,809**, Precision 0,534, **F1 0,643**.
- **Costo atteso** ridotto del **42%** rispetto alla persistenza e del **17%** rispetto alla soglia di default 0,5.
- **Ablazione.** Recupero di `Sunshine`/`Cloud3pm` e riparametrizzazione migliorano tutti i modelli. I guadagni sono però modesti, dello stesso ordine della variabilità fra i fold.
- **Strategie per lo sbilanciamento** (regressione logistica, ultime 20 000 righe del training): nessun trattamento, `class_weight`, SMOTE, NearMiss+SMOTE, LDA, **MetaCost** e MetaCost+LDA. Ripesatura, SMOTE e MetaCost sono equivalenti. Il modello finale incorpora l'asimmetria dei costi nella calibrazione della soglia.
- **Leakage temporale quantificato.** Uno split casuale sovrastima la PR-AUC di 2,3–3,1 punti, più dell'intero guadagno dello studio di ablazione.
- **Permutation importance.** Dominano l'umidità pomeridiana, le raffiche di vento e la pressione.

### 3. Clustering: regimi meteorologici

`clustering.ipynb` individua in modo non supervisionato i regimi meteorologici a partire da 12 variabili continue (campione casuale di 10 000 giornate su 120 381 osservazioni complete, 44 località). Località, mese e pioggia non sono usati dagli algoritmi e servono solo a validare i gruppi.

- **Preprocessing.** `log1p` sulla pioggia e standardizzazione. La PCA spiega il **61,5%** della varianza con 2 componenti e supera l'80% con 4.
- **K-Means con k = 4.** Nessun indice interno seleziona k = 4: Silhouette e Calinski-Harabasz preferiscono k = 2, Davies-Bouldin k = 3. La scelta segue una regola fissata in anticipo, cioè il k più grande con partizione **stabile** e **non riducibile a una singola variabile**. La stabilità è 0,980 ± 0,006 a k = 4 e crolla a 0,865 ± 0,161 a k = 5.
- **Quattro regimi:** freddo e stabile · mite e secco · caldo · perturbato e ventoso. Sono coerenti con stagioni, piovosità e geografia: nel regime perturbato piove il giorno dopo in oltre il 50% dei casi.
- **DBSCAN.** Non produce una partizione utilizzabile, perché i dati formano un continuo senza discontinuità di densità (confermato anche dalla **t-SNE**). Usato come **rilevatore di anomalie** (ε = 1,5), isola il 19,5% delle giornate, in larga parte del regime perturbato (pioggia il giorno dopo nel 49,2% contro il 15,2%).
- **Validazione esterna** con ARI, V-measure, homogeneity e completeness rispetto a stagione, `RainTomorrow` e località, più una mappa geografica dei regimi.

### 4. Serie temporali: temperatura mensile a Sydney

`temporal_series/time_series.ipynb` modella la temperatura pomeridiana media mensile (`Temp3pm`) a **Sydney**: 113 mesi, da febbraio 2008 a giugno 2017. Gli ultimi 24 mesi formano il test e **tutte le scelte sono fatte sul solo training**.

- **Identificazione (Box-Jenkins).** Decomposizione additiva, test ADF/KPSS, ACF/PACF. Il modello proposto è **SARIMA(1,0,0)(0,1,1)₁₂ con deriva** (`trend="c"`), confermato per AIC (236,5) e da una ricerca a griglia su 36 combinazioni. La ricerca usa `statsmodels`, perché `pmdarima` non è compatibile con Python 3.13.
- **Diagnostica.** Residui a rumore bianco (Ljung-Box non significativo, esclusi i primi 12 residui di inizializzazione).
- **Validazione a origine mobile** (5 fold, previsioni a 12 mesi) per confrontare famiglie di modelli: SARIMA, AR(1) + dummy mensili + trend, AR(1) + Fourier + trend, SARIMA con esogene. I candidati sono equivalenti entro un errore standard e si sceglie il più parsimonioso, cioè il SARIMA.
- **Test.** **MAE 0,801 °C** contro 0,893 della baseline naïve stagionale (**−10,3%**), RMSE 0,967, bias quasi nullo (−0,024 °C).
- **Stagionalità di fatto deterministica.** Il coefficiente MA stagionale è ≈ −1 e il SARIMA commette gli stessi errori del modello con dummy mensili. A distinguere i modelli validi da quelli inadeguati è il termine di tendenza.
- **Variabili esogene** (umidità e pressione). Migliorano il MAE (0,801 → 0,665) **solo nello scenario "oracolo"**, in cui se ne conoscono i valori futuri. Dovendole prevedere, il MAE sale a 0,848 e non portano vantaggi: il guadagno apparente è un **artefatto di lookahead**, quantificato e dichiarato.
- **Previsione finale** a 12 mesi (luglio 2017 – giugno 2018) con intervalli di confidenza al 95%.

---

## Esecuzione

**Requisiti:** Python 3.10+ (i notebook sono stati eseguiti con Python 3.13).

Dipendenze principali:

```bash
pip install pandas numpy scikit-learn statsmodels xgboost imbalanced-learn matplotlib seaborn jupyter
```

Passi:

1. Clona il repository:
   ```bash
   git clone https://github.com/valeriac23/WeatherAUS_DataScience_Project.git
   cd WeatherAUS_DataScience_Project
   ```
2. Scarica `weatherAUS.csv` da Kaggle e mettilo nella cartella `dataset/` nella root del progetto. I notebook la cercano automaticamente risalendo dalla cartella corrente.
3. Avvia Jupyter ed esegui i notebook nell'ordine consigliato:
   1. `eda_data_analysis.ipynb`
   2. i quattro notebook di classificazione, nell'ordine A → B → C → D (ciascuno scrive la propria riga in `ablation.json`, e il notebook D mostra il confronto finale)
   3. `clustering.ipynb`
   4. `time_series.ipynb`
   ```bash
   jupyter notebook
   ```

Tutti i risultati sono riproducibili: il seed è fissato (`RANDOM_STATE = 42`) in ogni notebook, incluso il bootstrap di MetaCost.

---

## Autori

- **Valeria Cannone**
- **Giada Remedia**
- **Alessandro Pettinaro**

Università Politecnica delle Marche — A.A. 2025/2026
