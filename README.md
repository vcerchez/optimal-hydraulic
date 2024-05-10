# optimal-hydraulic

**Objectif de l'exercice :** Construire un modèle ML pour prédire si la condition valve est 
optimale (=100%) ou non, pour chaque cycle.

Vous utiliserez les 2000 premiers cycles pour construire le modèle et le reste comme échantillon 
de test final.   

**Données :** Les données sont disponibles dans le dossier **/data**. Les données sont décrites 
dans le lien suivant : https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems.

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

**Bonus :** Voici quelques points bonus que vous pouvez choisir de réaliser en plus

*  Mettre votre solution sur github ou gitlab.
*  Ajouter des tests unitaires.
*  Containeriser votre code pour qu'il être exécuté facilement par un tiers.
*  Mettre en place une application web qui donne la prédiction pour un numéro de cycle donné en 
entrée.

# How to use it

For a quick demonstration one can pull a Docker image containing a pretrained model from the
[vcerchez/optimal-hydraulic](https://hub.docker.com/repository/docker/vcerchez/optimal-hydraulic/tags) 
repo.

The model run in the contaner can be queried through its API with its entry point at 
[localhost:8000/](http://localhost:8000/). For the API documentation, check 
[localhost:8000/docs](http://localhost:8000/docs).

# Contents of the repository

```
├── Dockerfile              : create image with the containerized ML model
├── LICENSE
├── README.md
├── data
│   │
│   └── data.7z             : archive with raw data
├── data_preparation.py     : data preparation for the model.ipynb notebook
├── data_transformation.py  : data preparation as sklearn transformers
├── eda.ipynb               : inital exoloratory data analysis
├── environment.yml         : project's conda environment
├── main.py                 : FastAPI app serving pretrained ML model
├── model.ipynb             : creation and and optimization of the ML model
├── requirements.txt        : requirements for the containerized ML model
├── test_api.ipynb          : notebook for testing API in container
├── test_app.py             : tests
└── train_model.py          : model training and pickling from the raw data
```
