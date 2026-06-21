# Food Volume & Portion Estimation

Streamlit-App zur Schaetzung von Essensmengen aus Bildern. Das Projekt
konzentriert sich auf **Volumen, Gramm und Naehrwerte**; Segmentierung und Food
Recognition kommen aus vortrainierten Modellen.

## Aktueller Ansatz

Die App hat zwei getrennte Mengenpfade:

| Situation | Methode | Ergebnis |
|---|---|---|
| Top- und Seitenbild vorhanden | Top-/Side-Silhouetten -> elliptische Querschnitte -> `volume_ml` | formbewusste Volumenschaetzung |
| Nur Top-Bild vorhanden | `mass_per_cm2[class] * area_cm2` | Gramm-Fallback, kein echtes Volumen |

Das ist wichtig: Ohne Seitenprofil gibt es keine belastbare Volumenschaetzung. Die App
bleibt dann nutzbar, markiert den Pfad aber fachlich als Fallback.

## Schnellstart

```bash
python setup_env.py
source .venv/bin/activate
streamlit run app.py
```

Tests:

```bash
.venv/bin/pytest
```

Training in Jupyter:

```bash
jupyter lab notebooks/
```

Das wichtigste Notebook ist `notebooks/01_training.ipynb`. Die eigentliche
Trainingslogik liegt in `foodvol/training.py`, damit sie testbar und
wiederverwendbar bleibt.

## Daten und Modelle

| Quelle / Modell | Rolle |
|---|---|
| FastSAM | vortrainierte Segmentierung |
| CLIP | vortrainierte Food Recognition und Non-Food-Filter |
| ECUSTFD | Training/Evaluation des Volumenmodells, weil Top-View, Side-View und Volumen vorhanden sind |
| Nutrition5k subset | Kalibrierung und Test des Top-Down-Gramm-Fallbacks |
| `foodvol/data/nutrition_db.csv` | Dichte, kcal, Makros und Portion-Priors |

Downloads:

```bash
python data/download_ecustfd.py
python data/download_nutrition5k.py
```

## Volumenpfad und trainierter Fallback

Im Live-Pfad werden die echte Top-Maske und Seiten-Maske entlang ihrer Laengsachse
ausgerichtet. Pro Position entsteht aus Top-Tiefe und Seiten-Hoehe ein elliptischer
Querschnitt; deren Integral ergibt das Volumen. Dadurch wird ein runder Apfel nicht
wie sein Begrenzungsquader behandelt.

Wenn die Silhouetten nicht ausgerichtet werden koennen, bleibt das trainierte Modell
als Fallback aktiv:

```text
artifact: artifacts/volume_model_trained.joblib
kind:     GradientBoostingRegressor
features: area_cm2, height_cm, area_x_height
```

Letzter reproduzierter ECUSTFD-Lauf:

| Split | MAPE | Bedeutung |
|---|---:|---|
| KFold | 19.1 % | neue Portionen aus aehnlichen Klassen |
| GroupKFold by food type | 26.2 % | strengere Schaetzung fuer neue Food-Klassen |

Das Training ist schnell, weil nicht das Bildmodell trainiert wird. Die Bilder
werden vorher in tabellarische Features umgewandelt; trainiert wird nur ein
kleines Regressionsmodell.

## App-Einstellungen

| Einstellung | Zweck |
|---|---|
| Detection detail | Grob steuert, wie viele Masken die Segmentierung zulaesst |
| Food filter | Steuert, wie streng unsichere Food-Kandidaten entfernt werden |

Die Skalierung hat keinen manuellen App-Regler. Sie kombiniert automatisch eine
erkannte 2-cm-Quadratreferenz, einen erkannten Standardteller, den Food-
Groessen-Prior und die Breitenkorrespondenz zwischen Top- und Seitenbild. Stark
widerspruechliche, schwache Hinweise werden als Ausreisser verworfen.

Details stehen in `ARCHITECTURE.md` und `TRAINING.md`.

## Repo-Hygiene

Die Tests verhindern, dass das Git-Repo versehentlich zu gross wird:

* getrackte Dateien muessen unter 500 MiB bleiben,
* ECUSTFD-Rohdaten gehoeren nicht ins Git,
* grosse Modellgewichte wie `.pt`, `.pth`, `.onnx`, `.ckpt` gehoeren nicht ins
  Git.

Aktuell liegt der getrackte Payload bei ca. 163 MiB.
