# Architektur

Die Pipeline beantwortet eine Kernfrage:

```text
Bildpixel -> Flaeche/Hoehe -> Volumen oder Gramm -> Naehrwerte
```

## Datenfluss

```text
Top image
  -> segmentation
  -> food recognition
  -> scale estimation
  -> area_cm2

Optional side image
  -> side segmentation
  -> height_cm

area_cm2 + height_cm
  -> volume model
  -> volume_ml
  -> density lookup
  -> mass_g, kcal, macros
```

Wenn `height_cm` fehlt:

```text
area_cm2 + food_class -> mass_per_cm2 prior -> mass_g
```

Dieser zweite Pfad ist ein Gramm-Fallback, keine Volumenschaetzung.

## Module

| Datei | Verantwortung |
|---|---|
| `foodvol/pipeline.py` | Orchestriert App/API-Schaetzungen |
| `foodvol/segmentation.py` | FastSAM plus klassischer Fallback fuer Masken |
| `foodvol/recognition.py` | CLIP Food Recognition und Non-Food-Gate |
| `foodvol/chessboard.py` | metrische Referenz, falls sichtbar |
| `foodvol/volume.py` | `VolumeEstimator`, Features, Artefakte, Metriken |
| `foodvol/portion.py` | entscheidet Volumenmodell vs. Mass-Prior |
| `foodvol/training.py` | reproduzierbares Training und Nutrition5k-Auswertung |
| `foodvol/nutrition.py` | Dichte, Portion-Priors, kcal und Makros |

## Skalierung

Die Pipeline braucht `cm/px`, um Pixel in reale Flaeche umzuwandeln.

| Modus | Verhalten |
|---|---|
| Auto | nutzt Referenz, wenn gefunden; sonst Food-Groessen-Prior |
| Food-size prior | ignoriert Referenz und nutzt typische Klassengroesse |
| Metric reference | bevorzugt sichtbare Referenz; faellt mit Hinweis zurueck |

Der Referenzwert in der App ist nur relevant, wenn wirklich ein Chessboard oder
eine vergleichbare metrische Referenz sichtbar ist.

## App-Regler

| Regler | Effekt |
|---|---|
| Detection detail | mehr/weniger Masken, hilfreich bei hellen Speisen oder unruhigem Hintergrund |
| Food filter | toleranter oder strenger CLIP-Food-Filter |
| Scale | Auswahl zwischen Auto, Food-Prior und metrischer Referenz |

Die Regler veraendern echte Pipeline-Parameter. Sie sind bewusst wenige und
allgemein gehalten, nicht auf einen einzelnen Hintergrundtyp optimiert.

## Grenzen

* Ohne Seitenbild fehlt echte Hoehe.
* ECUSTFD ist klein und nicht auf alle Zielgerichte repraesentativ.
* Nutrition5k hilft fuer Gramm-Priors, aber nicht fuer Volumentraining.
* Food-size scaling ist ein Prior und bleibt unsicherer als eine echte Referenz.

Der beste Qualitaetssprung waere ein eigener Datensatz mit Zielgerichten,
Top-/Side-Fotos und gewogener Ground Truth.
