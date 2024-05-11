# optimal-hydraulic

# Objectif du projet

Construire un modèle ML pour prédire si la condition valve est optimale (=100%) ou non, pour 
chaque cycle.

# Quick start

Pour une démonstration rapide, on peut extraire une image Docker contenant un modèle 
pré-entraîné: 
[vcerchez/optimal-hydraulic](https://hub.docker.com/repository/docker/vcerchez/optimal-hydraulic/tags). 

Le modèle s'exécute dans le conteneur et peut être interrogé via son API, dont le point d'entrée 
se trouve à [localhost:8000/](http://localhost:8000/). Pour la documentation de l'API, consultez 
[localhost:8000/docs](http://localhost:8000/docs).

# Données

Les données sont disponibles dans le dossier **/data**. Les données sont décrites dans le lien 
suivant : https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems.

Extrait du lien ci-dessus :

***************************

The data set contains raw process sensor data (i.e. without feature extraction) which are 
structured as matrices (**tab-delimited**) with the rows representing the cycles and the columns 
the data points within a cycle. The sensors involved are:

| Sensor | Physical quantity            | Unit  | Sampling rate |
|--------|------------------------------|-------|---------------|
| PS1    | Pressure                     | bar   | 100 Hz        |
| PS2    | Pressure                     | bar   | 100 Hz        |
| PS3    | Pressure                     | bar   | 100 Hz        |
| PS4    | Pressure                     | bar   | 100 Hz        |
| PS5    | Pressure                     | bar   | 100 Hz        |
| PS6    | Pressure                     | bar   | 100 Hz        |
| EPS1   | Motor power                  | W     | 100 Hz        |
| FS1    | Volume flow                  | l/min | 10 Hz         |
| FS2    | Volume flow                  | l/min | 10 Hz         |
| TS1    | Temperature                  | °C    | 1 Hz          |
| TS2    | Temperature                  | °C    | 1 Hz          |
| TS3    | Temperature                  | °C    | 1 Hz          |
| TS4    | Temperature                  | °C    | 1 Hz          |
| VS1    | Vibration                    | mm/s  | 1 Hz          |
| CE     | Cooling efficiency (virtual) | %     | 1 Hz          |
| CP     | Cooling power (virtual)      | kW    | 1 Hz          |
| SE     | Efficiency factor            | %     | 1 Hz          |

The target condition values are cycle-wise annotated in **profile.txt** (**tab-delimited**). As 
before, the row number represents the cycle number. The columns are


| Cooler condition (%)      | Valve condition (%)             | Internal pump leakage: | Hydraulic accumulator (bar)    | stable flag                                          |
|---------------------------|---------------------------------|------------------------|--------------------------------|------------------------------------------------------|
| 3: close to total failure | 100: optimal switching behavior | 0: no leakage          | 130: optimal pressure          | 0: conditions were stable                            |
| 20: reduced effifiency    | 90: small lag                   | 1: weak leakage        | 115: slightly reduced pressure | 1: static conditions might not have been reached yet |
| 100: full efficiency      | 80: severe lag                  | 2: severe leakage      | 100: severely reduced pressure |                                                      |
|                           | 73: close to total failure      |                        | 90: close to total failure     |                                                      |

***************************

Le dossier contient uniquement les 3 fichiers suivants, chaque ligne représentant un cycle : 
 
* PS2 (Pression (bar) echantillonnage 100Hz) 
* FS1 (Volume flow (l/min) echantillonnage 10Hz) 
* Profile : Fichier avec les variables dont la "valve condition" qui nous intéresse.

On utilise les 2000 premiers cycles pour construire le modèle et le reste comme échantillon de 
test final.

# Bonus

Quelques points bonus:

*  Mettre votre solution sur github ou gitlab.
*  Ajouter des tests unitaires.
*  Containeriser votre code pour qu'il être exécuté facilement par un tiers.
*  Mettre en place une application web qui donne la prédiction pour un numéro de cycle donné en 
entrée.

# Repository contents

```
├── LICENSE
├── README.md
├── app
│   ├── Dockerfile              : create image with the containerized ML model
│   ├── data_transformation.py  : data preparation as sklearn transformers
│   ├── main.py                 : FastAPI app serving pretrained ML model
│   ├── requirements.txt        : requirements for the containerized ML model
│   ├── test_api.ipynb          : notebook for testing model API in container
│   ├── test_app.py             : tests
│   └── train_model.py          : model training and pickling from the raw data
├── data
│   └── data.7z                 : archive with raw data
├── development
│   ├── data_preparation.py     : data preparation for the model.ipynb notebook
│   ├── eda.ipynb               : inital exoloratory data analysis
│   └── model.ipynb             : development and and optimization of the ML model
└── environment.yml             : conda environment
```
