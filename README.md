# velo

trajet salade ponsan -> supaero, la station est tjrs vide quand j'arrive...

collecteur sur le raspberry (autre repo) qui relève tout le réseau toutes les 15 min, sept -> nov 2024, 383 stations

```
uv sync
cp ../velotoulouse-collecteur/data/collected_*.parquet data/raw/
make data
make model   (long, ~4min)
make figures
make rain
```

## resultats (3 mois)

matin : ok jusqu'à 7h15 à salade ponsan - coteaux, apres aller à narbonne - sahuque
soir : partir avant 17h !!

brier 30 min, mes 7 stations :
profil 0.179
persistance 0.098
logit 0.097
gbm 0.092

pluie : -17% pendant l'heure, -15% l'heure d'apres, placebo ok (pas d'effet avant)

![](figures/trajet.png)
![](figures/pluie.png)

TODO
- app pour le trajet
- velos HS / elec
- places libres à l'arrivée
- camions de rééquilibrage ?
