# velo

trajet salade ponsan -> supaero, la station est tjrs vide quand j'arrive...

collecteur sur le raspberry (autre repo) qui relève tout le réseau toutes les 15 min depuis debut septembre

pour l'instant :
- grille 15 min
- meteo blagnac + vacances + feriés
- figures : proba >= 2 vélos sur mon trajet, heatmaps, carte du matin

```
uv sync
cp ../velotoulouse-collecteur/data/collected_*.parquet data/raw/
make data
make figures
```

TODO
- predire s'il reste un velo quand j'arrive ??
- pluie
- nettoyer figures.py
