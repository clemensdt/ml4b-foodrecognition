# Training

Dieses Projekt trainiert nur die Mengenlogik, nicht die Bild-Basismodelle.

| Baustein | Quelle | Wird selbst trainiert? |
|---|---|---:|
| Segmentierung | FastSAM | nein |
| Food Recognition | CLIP | nein |
| Volumenmodell | ECUSTFD-Features | ja |
| Gramm-Fallback | Nutrition5k-Priors | kalibriert, nicht als Bildmodell trainiert |

## Volumentraining

Das echte Volumenmodell braucht eine Flaeche aus dem Top-Bild und eine Hoehe aus
dem Seitenbild:

```text
area_cm2, height_cm, area_cm2 * height_cm -> volume_ml
```

ECUSTFD ist dafuer der passende Datensatz, weil er Top-View, Side-View,
metrische Referenz, gemessenes Gewicht und gemessenes Volumen enthaelt.

Die App laedt standardmaessig:

```text
artifacts/volume_model_trained.joblib
```

Aktuelles Deployment-Modell:

```text
model_kind = gbr
features   = area_cm2, height_cm, area_x_height
```

## Nutrition5k

Nutrition5k wird nicht fuer Volumentraining verwendet. Es fehlen Seitenhoehe und
Volumen-Ground-Truth. Der Datensatz ist trotzdem nuetzlich, weil er Top-Down-
Bilder und gewogene Lebensmittel enthaelt.

Wir nutzen ihn fuer den Fallback:

```text
mass_g = mass_per_cm2[class] * area_cm2
```

`foodvol.training.derive_n5k_mass_priors(...)` erzeugt robuste Median-Priors pro
Klasse. Die Tests vergleichen diese Priors mit
`data/n5k_meta/n5k_class_priors.json`, damit die Fallback-Grammwerte
reproduzierbar bleiben.

## Training ausfuehren

Empfohlen:

```bash
jupyter lab notebooks/01_training.ipynb
```

Oder programmgesteuert:

```python
from foodvol import config, training

df = training.load_ecustfd_features()
leaderboard = training.score_volume_models(df)
estimator = training.fit_final_volume_model(
    df,
    metrics={"leaderboard": leaderboard.to_dict(orient="records")},
    save_path=config.VOLUME_MODEL_PATH,
)
```

Danach nutzt die App automatisch das gespeicherte Artefakt.

## Ergebnisse interpretieren

Letzter reproduzierter Lauf:

| Split | MAPE | Interpretation |
|---|---:|---|
| KFold | 19.1 % | Performance auf zufaelligen, aehnlichen Portionen |
| GroupKFold by food type | 26.2 % | strengere Schaetzung fuer neue Food-Klassen |

Das Training dauert nur kurz, weil die Bildverarbeitung vorher in gecachte
Features geschrieben wird. Das Problem ist daher nicht zu wenig Trainingszeit,
sondern zu wenig passende Ground-Truth-Daten fuer eure Zielgerichte.

## Was das Modell besser macht

Mehr Daten helfen mehr als ein groesseres Modell. Der beste Ausbaupfad:

1. Zielgerichte definieren.
2. Pro Portion Top- und Seitenfoto aufnehmen.
3. Gewicht mit Kuechenwaage erfassen.
4. Wenn moeglich Volumen messen oder standardisiert ableiten.
5. Daten in dieselbe Feature-Struktur wie ECUSTFD bringen.
6. `notebooks/01_training.ipynb` erneut ausfuehren.

Nutrition5k verbessert den Gramm-Fallback. Fuer echtes Volumen ersetzt es keinen
Top-/Side-Datensatz.
