# uv run streamlit run app/app.py

import numpy as np
import pandas as pd
import streamlit as st

from velo import live
from velo.config import HOME, HOME_COORDS, MIN_BIKES, SCHOOL, SCHOOL_COORDS, TZ
from velo.features import HORIZONS

st.set_page_config(page_title="Vais-je trouver un vélo ?")


SAFE = 0.9  # au dessus de ça on considère que c'est bon


def distance(lat, lon, origin):
    # à vol d'oiseau, en m. approx plate, suffisant à cette échelle
    dx = np.radians(lon - origin[1]) * np.cos(np.radians(origin[0]))
    dy = np.radians(lat - origin[0])
    return 6_371_000 * np.hypot(dx, dy)


@st.cache_resource
def models():
    return live.load_models()


@st.cache_data(ttl=60)
def snapshot():
    now = pd.Timestamp.now(tz=TZ)
    return now, live.fetch_status(), live.fetch_weather(now)


models_h, profile = models()
now, status, weather = snapshot()
pred = live.predict(status, now, weather, models_h, profile).set_index("station")

st.title("Vais-je trouver un vélo ?")
st.caption(f"État du réseau VélôToulouse à {now:%H:%M}, pluie sur la dernière heure : "
           f"{weather['rain'].iloc[0]:.1f} mm. Probabilité d'avoir au moins {MIN_BIKES} vélos "
           "à la station quand j'y arrive.")

direction = st.radio("Trajet", ["Maison → SUPAERO", "SUPAERO → Maison"], horizontal=True)
start, end, origin = ((HOME, SCHOOL, HOME_COORDS) if direction.startswith("Maison")
                      else (SCHOOL, HOME, SCHOOL_COORDS))
horizon = st.select_slider("J'arrive à la station dans", HORIZONS, value=15, format_func=lambda h: f"{h} min")

# des numéros de station peuvent disparaitre du flux
start, end = ({k: v for k, v in d.items() if k in pred.index} for d in (start, end))
if not start:
    st.error("Aucune station de départ du trajet n'est présente dans le flux temps réel.")
    st.stop()
table = pred.loc[list(start), ["lat", "lon", "bikes", "docks"] + [f"p{h}" for h in HORIZONS]]
table.insert(0, "distance", distance(table["lat"], table["lon"], origin).round(-1).astype(int))
table = table.drop(columns=["lat", "lon"]).sort_values("distance")
table.index = [start[i] for i in table.index]
p = table[f"p{horizon}"]

st.subheader("Départ")
if p.notna().any():
    best = p[p >= SAFE].index[0] if (p >= SAFE).any() else p.idxmax()
    st.markdown(f"Station conseillée : **{best}** ({table.loc[best, 'distance']} m), "
                f"{100 * p[best]:.0f} % de chances d'y trouver au moins {MIN_BIKES} vélos dans {horizon} min.")
else:
    st.warning("Toutes les stations de départ sont fermées.")
st.caption(f"Règle : la station la plus proche dont la probabilité atteint {SAFE:.0%}, sinon la plus probable. "
           "Une station fermée n'a pas de probabilité.")
shown = table.copy()
shown[[f"p{h}" for h in HORIZONS]] *= 100  # ProgressColumn a pas de mode pourcentage, que du printf
st.dataframe(
    shown.rename(columns={"distance": "distance (m)", "bikes": "vélos", "docks": "places",
                          **{f"p{h}": f"dans {h} min" for h in HORIZONS}}),
    column_config={f"dans {h} min": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100)
                   for h in HORIZONS},
)

st.subheader("Arrivée")
# TODO prédire les places aussi, pour l'instant juste l'état actuel
arrival = pred.loc[list(end), ["docks"]]
arrival.index = [end[i] for i in arrival.index]
st.dataframe(arrival.rename(columns={"docks": "places libres maintenant"}))
st.caption("Le modèle prédit les vélos, pas les places : à l'arrivée, seul l'état actuel est affiché.")

st.subheader("N'importe quelle station")
names = status.set_index("station")["name"].sort_values()
default = next(iter(HOME))
choice = st.selectbox("Station", names.index, format_func=names.get,
                      index=list(names.index).index(default) if default in names.index else 0)
row = pred.loc[choice]
cols = st.columns(len(HORIZONS) + 1)
cols[0].metric("Vélos maintenant", int(row["bikes"]))
for col, h in zip(cols[1:], HORIZONS):
    col.metric(f"P(≥ {MIN_BIKES}) dans {h} min", "fermée" if pd.isna(row[f"p{h}"]) else f"{100 * row[f'p{h}']:.0f} %")

if choice not in profile.index.get_level_values(0):
    st.caption("Pas de profil habituel : station absente de la période étudiée.")
    st.stop()
curve = profile.loc[choice].unstack(0).rename(columns={"work": "jour ouvré", "off": "week-end / férié"})
curve.index = [f"{s // 4:02d}:{15 * (s % 4):02d}" for s in curve.index]
st.markdown(f"Profil habituel (sept. - oct. 2024) : part du temps avec au moins {MIN_BIKES} vélos")
st.line_chart(curve, y_label="probabilité", x_label="heure")
