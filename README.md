# Dashboard Meta + Zoho — versión para compartir

Esta carpeta contiene la versión actualizada del dashboard en Streamlit.

## Cambios incluidos

- El filtro de `Fuente de Posible cliente` acepta únicamente `Facebook Ads` y `FacebookAds`.
- Se elimina `fb`.
- Lead Journey incorpora un funnel general con la conversión **simple** de cada etapa frente a la anterior.
- Se incorpora una tabla general con:
  - número de leads por etapa;
  - conversión simple;
  - conversión acumulada desde Registros.
- Se conserva el funnel por anuncio debajo del funnel general.
- Los IDs de Google Drive ya no se escriben dentro del código; se cargan desde Streamlit Secrets o variables de entorno.

## Archivos

- `app.py`: dashboard.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: configuración básica.
- `secrets.toml.example`: plantilla para configurar los IDs sin publicarlos.

## Probar localmente

1. Instala Python 3.11 o 3.12.
2. Crea un entorno virtual.
3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Crea la carpeta `.streamlit` junto a `app.py`.
5. Copia `secrets.toml.example` como `.streamlit/secrets.toml`.
6. Sustituye cada `PEGA_AQUI_EL_ID` por el ID real de su CSV de Google Drive.
7. Asegúrate de que cada CSV tenga permiso de lectura mediante enlace.
8. Ejecuta:

```bash
streamlit run app.py
```

## Publicar para el equipo con Streamlit Community Cloud

1. Crea un repositorio privado o público en GitHub.
2. Sube:
   - `app.py`
   - `requirements.txt`
   - la carpeta `.streamlit/config.toml`
3. **No subas** `.streamlit/secrets.toml`.
4. En Streamlit Community Cloud crea una app desde ese repositorio y selecciona `app.py`.
5. En **App settings → Secrets**, pega:

```toml
ID_LISTA_REGISTROS = "..."
ID_DT_LEADS = "..."
ID_META = "..."
ID_209_VIDEO = "..."
ID_209_ESTATICO = "..."
ID_LT_V1 = "..."
```

6. Guarda y reinicia la app. Streamlit te dará una URL que puedes compartir con el equipo.

## Nota metodológica sobre En proceso → Cierre

El código actual clasifica `En proceso de venta` usando el `Stage` actual y clasifica `Cierre` como `Contrato Firmado - Cierre Logrado`. Por eso un lead cerrado deja de formar parte de los stages activos de `En proceso de venta`.

La tabla calcula la conversión simple solicitada como:

`leads de la etapa actual / leads de la etapa anterior`

y la acumulada como:

`leads de la etapa actual / Registros`

Esto es correcto como lectura del snapshot actual. Para una tasa histórica estricta de transición `En proceso → Cierre` por cohorte, se necesitaría historial de cambios de Stage por lead.
