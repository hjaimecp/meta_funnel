
import os
import re
from decimal import Decimal, InvalidOperation
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Meta",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Meta")


# ============================================================
# 1. GOOGLE DRIVE
# ============================================================

# Pega únicamente el ID de cada archivo CSV de Google Drive.
#
# Ejemplo:
# https://drive.google.com/file/d/1ABCxyz123/view?usp=sharing
# ID = 1ABCxyz123

def get_config_value(name):
    """
    Lee primero Streamlit Secrets y, si no existe, una variable de entorno.
    Así los IDs no quedan expuestos en el repositorio.
    """
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if not value:
        value = os.getenv(name, "")

    return str(value).strip()


ID_LISTA_REGISTROS = get_config_value("ID_LISTA_REGISTROS")
ID_DT_LEADS = get_config_value("ID_DT_LEADS")
ID_META = get_config_value("ID_META")

# Nuevos CSV de formularios
ID_209_VIDEO = get_config_value("ID_209_VIDEO")
ID_209_ESTATICO = get_config_value("ID_209_ESTATICO")
ID_LT_V1 = get_config_value("ID_LT_V1")

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def clean_colnames(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_id(value):
    """
    Homogeneiza IDs largos para cruces Meta <-> Zoho.
    """
    if pd.isna(value):
        return pd.NA

    s = str(value).strip()

    if s == "":
        return pd.NA

    if s.startswith("'"):
        s = s[1:].strip()

    if re.fullmatch(r"\d+", s):
        return s

    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".")[0]

    try:
        d = Decimal(s)

        if d == d.to_integral_value():
            return format(d.quantize(Decimal("1")), "f")

    except (InvalidOperation, ValueError):
        pass

    return s


def normalize_email(value):
    if pd.isna(value):
        return pd.NA

    s = str(value).strip().lower()

    return s if s else pd.NA


def norm_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    ).casefold()


def to_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def to_money(series):
    s = series.astype(str)

    s = s.str.replace(
        r"[^\d,\.\-]",
        "",
        regex=True
    )

    both = (
        s.str.contains(",", na=False)
        & s.str.contains(r"\.", na=False)
    )

    s.loc[both] = (
        s.loc[both]
        .str.replace(",", "", regex=False)
    )

    only_comma = (
        s.str.contains(",", na=False)
        & ~s.str.contains(r"\.", na=False)
    )

    s.loc[only_comma] = (
        s.loc[only_comma]
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0)


def safe_div(num, den, mult=1):
    if den is None or pd.isna(den) or den == 0:
        return np.nan

    return (num / den) * mult


def money(value):
    if pd.isna(value):
        return "—"

    return f"${value:,.2f}"


def pct(value):
    if pd.isna(value):
        return "—"

    return f"{value:,.2f}%"


def integer(value):
    if pd.isna(value):
        return "—"

    return f"{int(round(value)):,}"


def require_col(df, column, dataset):
    if column not in df.columns:
        st.error(
            f'Falta la columna "{column}" en {dataset}.'
        )
        st.stop()


def first_existing_col(df, candidates):
    """Devuelve la primera columna existente entre varias alternativas."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ============================================================
# 3. DESCARGA DIRECTA DE CSV DESDE GOOGLE DRIVE
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def cargar_csv_drive(file_id):

    if (
        not file_id
        or file_id.startswith("PEGA_AQUI")
    ):
        raise ValueError(
            "Falta configurar uno de los IDs de Google Drive."
        )

    url = (
        "https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    response = requests.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("Content-Type", "")
        .lower()
    )

    if "text/html" in content_type:
        raise ValueError(
            "Google Drive devolvió HTML en lugar del CSV. "
            "Verifica que el archivo tenga acceso de lectura mediante enlace."
        )

    return pd.read_csv(
        BytesIO(response.content),
        sep=None,
        engine="python",
        dtype=str,
        encoding="utf-8-sig"
    )


# ============================================================
# 4. CARGA DE LOS DATASETS
# ============================================================

with st.spinner(
    "Cargando datos desde Google Drive..."
):

    try:

        lista_registros = clean_colnames(
            cargar_csv_drive(
                ID_LISTA_REGISTROS
            )
        )

        DT_Leads = clean_colnames(
            cargar_csv_drive(
                ID_DT_LEADS
            )
        )

        meta = clean_colnames(
            cargar_csv_drive(
                ID_META
            )
        )

        form_209_video = clean_colnames(
            cargar_csv_drive(
                ID_209_VIDEO
            )
        )

        form_209_estatico = clean_colnames(
            cargar_csv_drive(
                ID_209_ESTATICO
            )
        )

        form_lt_v1 = clean_colnames(
            cargar_csv_drive(
                ID_LT_V1
            )
        )

    except Exception as e:

        st.error(
            "No fue posible cargar los archivos desde Google Drive."
        )

        st.exception(e)
        st.stop()


# Evita duplicados EXACTOS accidentales del archivo.
# No elimina filas legítimas con diferentes desgloses.
lista_registros = lista_registros.drop_duplicates().copy()
DT_Leads = DT_Leads.drop_duplicates().copy()
meta = meta.drop_duplicates().copy()
form_209_video = form_209_video.drop_duplicates().copy()
form_209_estatico = form_209_estatico.drop_duplicates().copy()
form_lt_v1 = form_lt_v1.drop_duplicates().copy()


# ============================================================
# 5. VALIDACIÓN DE COLUMNAS
# ============================================================

required_lista = [
    "Fecha de creación",
    "Correo electrónico"
]

required_leads = [
    "Hora de creación",
    "Fecha de calificación",
    "Nombre completo",
    "Fuente de Posible cliente",
    "utm_content",
    "Tamaño de la empresa",
    "Estado de Posible cliente",
    "Estatus de lead",
    "se ha convertido",
    "Stage",
    "Correo electrónico",
    "Importe",
    "Fecha de cierre"
]

required_meta = [
    "Identificador del anuncio",
    "Nombre del anuncio",
    "Día",
    "Plataforma",
    "Ubicación",
    "Importe gastado (MXN)",
    "Resultados",
    "Alcance",
    "Impresiones",
    "Clics únicos en el enlace"
]

for col in required_lista:
    require_col(
        lista_registros,
        col,
        "lista_registros"
    )

for col in required_leads:
    require_col(
        DT_Leads,
        col,
        "DT_Leads"
    )

for col in required_meta:
    require_col(
        meta,
        col,
        "meta"
    )


# ============================================================
# 6. FECHAS
# ============================================================

# Meta:
# El CSV puede traer fechas en más de un formato.
# Se intenta primero DD/MM/YYYY (formato habitual del reporte de Meta)
# y después variantes ISO para evitar que julio se convierta en NaT.

def parse_meta_dates(series):
    s = (
        series
        .astype(str)
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA
            }
        )
    )

    result = pd.Series(
        pd.NaT,
        index=s.index,
        dtype="datetime64[ns]"
    )

    # 1) DD/MM/YYYY
    mask = result.isna() & s.notna()
    result.loc[mask] = pd.to_datetime(
        s.loc[mask],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # 2) YYYY-MM-DD
    mask = result.isna() & s.notna()
    result.loc[mask] = pd.to_datetime(
        s.loc[mask],
        format="%Y-%m-%d",
        errors="coerce"
    )

    # 3) DD-MM-YYYY
    mask = result.isna() & s.notna()
    result.loc[mask] = pd.to_datetime(
        s.loc[mask],
        format="%d-%m-%Y",
        errors="coerce"
    )

    # 4) Último intento para cualquier variante restante.
    mask = result.isna() & s.notna()
    result.loc[mask] = pd.to_datetime(
        s.loc[mask],
        errors="coerce",
        dayfirst=True
    )

    return result.dt.normalize()


meta["fecha"] = parse_meta_dates(
    meta["Día"]
)


# Zoho / DT_Leads:
# 12 Aug 2026
DT_Leads["fecha"] = pd.to_datetime(
    DT_Leads["Hora de creación"],
    errors="coerce",
    dayfirst=True
).dt.normalize()


DT_Leads["fecha_cierre"] = pd.to_datetime(
    DT_Leads["Fecha de cierre"],
    errors="coerce",
    dayfirst=True
).dt.normalize()


# lista_registros:
# ejemplo 08/10/2026 8:45am = Aug 10, 2026
lista_registros["fecha"] = pd.to_datetime(
    lista_registros["Fecha de creación"],
    errors="coerce",
    dayfirst=False
).dt.normalize()


# ============================================================
# 7. META: LIMPIEZA
# ============================================================

meta["ad_id"] = (
    meta["Identificador del anuncio"]
    .map(normalize_id)
)

meta["ad_name"] = (
    meta["Nombre del anuncio"]
    .astype(str)
    .str.strip()
)

for col in [
    "Importe gastado (MXN)",
    "Resultados",
    "Alcance",
    "Impresiones",
    "Clics únicos en el enlace"
]:
    meta[col] = to_numeric(
        meta[col]
    )


# ============================================================
# 8. MAPA ID -> NOMBRE DEL ANUNCIO
# ============================================================

ad_lookup = (
    meta.loc[
        meta["ad_id"].notna()
        & meta["ad_name"].notna(),
        [
            "ad_id",
            "ad_name"
        ]
    ]
    .drop_duplicates(
        subset=["ad_id"],
        keep="last"
    )
)

ad_map = dict(
    zip(
        ad_lookup["ad_id"],
        ad_lookup["ad_name"]
    )
)

name_to_ids = (
    ad_lookup
    .groupby("ad_name")["ad_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 9. DT_LEADS: LIMPIEZA Y CRUCE DE ANUNCIO
# ============================================================

DT_Leads["ad_id"] = (
    DT_Leads["utm_content"]
    .map(normalize_id)
)

DT_Leads["ad_name"] = (
    DT_Leads["ad_id"]
    .map(ad_map)
)

DT_Leads["ad_name"] = (
    DT_Leads["ad_name"]
    .fillna(
        DT_Leads["ad_id"].map(
            lambda x:
            f"ID sin nombre: {x}"
            if pd.notna(x)
            else "Sin anuncio"
        )
    )
)

DT_Leads["email_norm"] = (
    DT_Leads["Correo electrónico"]
    .map(normalize_email)
)

DT_Leads["importe_num"] = (
    to_money(
        DT_Leads["Importe"]
    )
)


# Conteo de toques de Perfilamiento. Si la columna no existe, se crea vacía
# para que el resto del dashboard no se rompa.
if "Conteo" in DT_Leads.columns:
    DT_Leads["conteo_num"] = pd.to_numeric(
        DT_Leads["Conteo"],
        errors="coerce"
    )
else:
    DT_Leads["conteo_num"] = np.nan

# Zoho puede exportar el identificador con distintos nombres.
LEAD_ID_COL = first_existing_col(
    DT_Leads,
    ["ID", "Id", "id", "ID de Posible cliente", "Posible cliente ID", "Lead ID"]
)


# ============================================================
# 10. FILTRO GENERAL DE FACEBOOK EN LEAD JOURNEY
# ============================================================

facebook_sources = {
    "facebook ads",
    "facebookads"
}

DT_Leads_fb = (
    DT_Leads[
        DT_Leads[
            "Fuente de Posible cliente"
        ]
        .map(norm_text)
        .isin(facebook_sources)
    ]
    .copy()
)


# ============================================================
# 11. LISTA_REGISTROS: CRUCE POR EMAIL
# ============================================================

lista_registros["email_norm"] = (
    lista_registros[
        "Correo electrónico"
    ]
    .map(normalize_email)
)

email_to_ad = (
    DT_Leads.loc[
        DT_Leads["email_norm"].notna(),
        [
            "email_norm",
            "ad_id",
            "ad_name"
        ]
    ]
    .drop_duplicates(
        subset=["email_norm"],
        keep="last"
    )
)

lista_registros = (
    lista_registros
    .merge(
        email_to_ad,
        on="email_norm",
        how="left"
    )
)

lista_registros["ad_name"] = (
    lista_registros["ad_name"]
    .fillna(
        "Sin anuncio identificado"
    )
)



# ============================================================
# 12. PREPARACIÓN DE LOS FORMULARIOS DE META
# ============================================================

FORM_START = pd.Timestamp("2026-07-10")
FORM_END = pd.Timestamp("2026-08-10")

QUESTION_COLS = [
    "¿cuántos_colaboradores_tiene_tu_empresa?",
    "¿cuál_es_la_necesidad_crítica_que_deseas_resolver?_",
    "¿cuántas_llamadas_recibe_tu_empresa_al_día?",
    "¿qué_presupuesto_mensual_consideras_para_una_solución_telefónica?",
    "¿qué_te_gustaría_hacer_hoy?",
    "¿cuándo_planeas_implementar_la_solución?",
    "conditional_question_1",
    "conditional_question_2",
    "¿cuántas_personas_necesitan_hacer_ó_recibir_llamadas?"
]

QUESTION_LABELS = {
    "¿cuántos_colaboradores_tiene_tu_empresa?": "Colaboradores",
    "¿cuál_es_la_necesidad_crítica_que_deseas_resolver?_": "Necesidad crítica",
    "¿cuántas_llamadas_recibe_tu_empresa_al_día?": "Llamadas al día",
    "¿qué_presupuesto_mensual_consideras_para_una_solución_telefónica?": "Presupuesto mensual",
    "¿qué_te_gustaría_hacer_hoy?": "Qué quiere hacer hoy",
    "¿cuándo_planeas_implementar_la_solución?": "Implementación",
    "conditional_question_1": "Medio de contacto",
    "conditional_question_2": "Necesidad específica",
    "¿cuántas_personas_necesitan_hacer_ó_recibir_llamadas?": "Personas que usan llamadas"
}


def clean_answer(value):
    """
    Hace legibles respuestas como:
    2_a_5_personas -> 2 a 5 personas
    """
    if pd.isna(value):
        return "Sin respuesta"

    s = str(value).strip()

    if s == "" or norm_text(s) in {"nan", "none"}:
        return "Sin respuesta"

    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    return s



# ============================================================
# NORMALIZACIÓN DE ESTADOS DE MÉXICO
# ============================================================

MEXICO_STATE_ALIASES = {
    # Aguascalientes
    "aguascalientes": "Aguascalientes",
    "ags": "Aguascalientes",

    # Baja California
    "baja california": "Baja California",
    "bc": "Baja California",

    # Baja California Sur
    "baja california sur": "Baja California Sur",
    "bcs": "Baja California Sur",

    # Campeche
    "campeche": "Campeche",

    # Chiapas
    "chiapas": "Chiapas",
    "chis": "Chiapas",

    # Chihuahua
    "chihuahua": "Chihuahua",
    "chih": "Chihuahua",

    # Ciudad de México
    "ciudad de mexico": "Ciudad de México",
    "cdmx": "Ciudad de México",
    "distrito federal": "Ciudad de México",
    "df": "Ciudad de México",

    # Coahuila
    "coahuila": "Coahuila",
    "coahuila de zaragoza": "Coahuila",
    "coah": "Coahuila",

    # Colima
    "colima": "Colima",

    # Durango
    "durango": "Durango",
    "dgo": "Durango",

    # Estado de México
    "estado de mexico": "Estado de México",
    "edo mex": "Estado de México",
    "edo de mex": "Estado de México",
    "edomex": "Estado de México",
    "estado de mexico y cdmx": "Estado de México",
    "mexico": "Estado de México",
    "mex": "Estado de México",
    "de mexico": "Estado de México",
    "coacalco": "Estado de México",
    "zumpango": "Estado de México",

    # Guanajuato
    "guanajuato": "Guanajuato",
    "gto": "Guanajuato",

    # Guerrero
    "guerrero": "Guerrero",
    "gro": "Guerrero",

    # Hidalgo
    "hidalgo": "Hidalgo",
    "hgo": "Hidalgo",

    # Jalisco
    "jalisco": "Jalisco",
    "jal": "Jalisco",

    # Michoacán
    "michoacan": "Michoacán",
    "michoacan de ocampo": "Michoacán",
    "mich": "Michoacán",

    # Morelos
    "morelos": "Morelos",
    "mor": "Morelos",

    # Nayarit
    "nayarit": "Nayarit",
    "nay": "Nayarit",

    # Nuevo León
    "nuevo leon": "Nuevo León",
    "nuevo nuevo leon": "Nuevo León",
    "nuevo le9n": "Nuevo León",
    "nl": "Nuevo León",
    "n l": "Nuevo León",
    "minterrey n l": "Nuevo León",
    "monterrey n l": "Nuevo León",
    "monterrey": "Nuevo León",

    # Oaxaca
    "oaxaca": "Oaxaca",
    "oax": "Oaxaca",

    # Puebla
    "puebla": "Puebla",
    "pue": "Puebla",

    # Querétaro
    "queretaro": "Querétaro",
    "queretaro arteaga": "Querétaro",
    "qro": "Querétaro",

    # Quintana Roo
    "quintana roo": "Quintana Roo",
    "q roo": "Quintana Roo",
    "qroo": "Quintana Roo",

    # San Luis Potosí
    "san luis potosi": "San Luis Potosí",
    "slp": "San Luis Potosí",

    # Sinaloa
    "sinaloa": "Sinaloa",
    "sin": "Sinaloa",

    # Sonora
    "sonora": "Sonora",
    "son": "Sonora",

    # Tabasco
    "tabasco": "Tabasco",
    "tab": "Tabasco",

    # Tamaulipas
    "tamaulipas": "Tamaulipas",
    "tamps": "Tamaulipas",
    "reynosa": "Tamaulipas",

    # Tlaxcala
    "tlaxcala": "Tlaxcala",
    "tlax": "Tlaxcala",

    # Veracruz
    "veracruz": "Veracruz",
    "veracruz de ignacio de la llave": "Veracruz",
    "ver": "Veracruz",

    # Yucatán
    "yucatan": "Yucatán",
    "yuc": "Yucatán",

    # Zacatecas
    "zacatecas": "Zacatecas",
    "zac": "Zacatecas"
}


def normalize_mexico_state(value):
    """
    Convierte variantes, abreviaturas y algunos municipios obvios
    a una de las 32 entidades federativas.

    Valores que claramente no son estados (emails, teléfonos,
    textos libres, números, países extranjeros) se dejan como NA.
    """
    if pd.isna(value):
        return pd.NA

    raw = str(value).strip()

    if raw == "":
        return pd.NA

    # Descartar valores evidentemente inválidos.
    if "@" in raw:
        return pd.NA

    if re.fullmatch(r"\d+", raw):
        return pd.NA

    # Normalización básica de texto.
    import unicodedata

    key = unicodedata.normalize(
        "NFKD",
        raw
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    key = key.lower()
    key = key.replace("_", " ")
    key = re.sub(r"[^a-z0-9\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()

    # Casos explícitamente no válidos / fuera de México.
    invalid_values = {
        "canada",
        "cotizacion",
        "me interesa el dispositivo para instalar en pick up"
    }

    if key in invalid_values:
        return pd.NA

    return MEXICO_STATE_ALIASES.get(
        key,
        pd.NA
    )


# Coordenadas aproximadas de centroides estatales.
# Se usan para un mapa de frecuencia con burbujas,
# evitando GeoJSON y dependencias adicionales.
MEXICO_STATE_CENTROIDS = {
    "Aguascalientes": (21.8853, -102.2916),
    "Baja California": (30.8406, -115.2838),
    "Baja California Sur": (26.0444, -111.6661),
    "Campeche": (19.8301, -90.5349),
    "Chiapas": (16.7569, -93.1292),
    "Chihuahua": (28.6330, -106.0691),
    "Ciudad de México": (19.4326, -99.1332),
    "Coahuila": (27.0587, -101.7068),
    "Colima": (19.2452, -103.7241),
    "Durango": (24.0277, -104.6532),
    "Estado de México": (19.2826, -99.6557),
    "Guanajuato": (21.0190, -101.2574),
    "Guerrero": (17.4392, -99.5451),
    "Hidalgo": (20.0911, -98.7624),
    "Jalisco": (20.6597, -103.3496),
    "Michoacán": (19.5665, -101.7068),
    "Morelos": (18.6813, -99.1013),
    "Nayarit": (21.7514, -104.8455),
    "Nuevo León": (25.5922, -99.9962),
    "Oaxaca": (17.0732, -96.7266),
    "Puebla": (19.0414, -98.2063),
    "Querétaro": (20.5888, -100.3899),
    "Quintana Roo": (19.1817, -88.4791),
    "San Luis Potosí": (22.1565, -100.9855),
    "Sinaloa": (24.8091, -107.3940),
    "Sonora": (29.2972, -110.3309),
    "Tabasco": (17.8409, -92.6189),
    "Tamaulipas": (24.2669, -98.8363),
    "Tlaxcala": (19.3139, -98.2404),
    "Veracruz": (19.1738, -96.1342),
    "Yucatán": (20.7099, -89.0943),
    "Zacatecas": (22.7709, -102.5833)
}




MEXICO_STATE_TO_GEOJSON = {
    "Aguascalientes": "Aguascalientes",
    "Baja California": "Baja California",
    "Baja California Sur": "Baja California Sur",
    "Campeche": "Campeche",
    "Chiapas": "Chiapas",
    "Chihuahua": "Chihuahua",
    "Ciudad de México": "Distrito Federal",
    "Coahuila": "Coahuila de Zaragoza",
    "Colima": "Colima",
    "Durango": "Durango",
    "Estado de México": "México",
    "Guanajuato": "Guanajuato",
    "Guerrero": "Guerrero",
    "Hidalgo": "Hidalgo",
    "Jalisco": "Jalisco",
    "Michoacán": "Michoacán de Ocampo",
    "Morelos": "Morelos",
    "Nayarit": "Nayarit",
    "Nuevo León": "Nuevo León",
    "Oaxaca": "Oaxaca",
    "Puebla": "Puebla",
    "Querétaro": "Querétaro",
    "Quintana Roo": "Quintana Roo",
    "San Luis Potosí": "San Luis Potosí",
    "Sinaloa": "Sinaloa",
    "Sonora": "Sonora",
    "Tabasco": "Tabasco",
    "Tamaulipas": "Tamaulipas",
    "Tlaxcala": "Tlaxcala",
    "Veracruz": "Veracruz de Ignacio de la Llave",
    "Yucatán": "Yucatán",
    "Zacatecas": "Zacatecas"
}


@st.cache_data(ttl=86400, show_spinner=False)
def cargar_geojson_mexico():
    """
    Carga límites estatales de México para el choropleth.
    No requiere geopandas ni paquetes adicionales.
    """
    url = (
        "https://raw.githubusercontent.com/"
        "strotgen/mexico-leaflet/master/states.geojson"
    )

    response = requests.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def prepare_form_df(df, source_name):
    """
    Homologa los tres CSV de formularios.
    """

    out = df.copy()

    required = [
        "created_time",
        "ad_id",
        "ad_name",
        "email",
        "state"
    ] + QUESTION_COLS

    for col in required:
        require_col(
            out,
            col,
            source_name
        )

    # Conservar fecha local escrita en el CSV.
    # Evitamos conversión UTC para no mover registros cercanos a medianoche.
    out["fecha_formulario"] = pd.to_datetime(
        out["created_time"]
        .astype(str)
        .str.slice(0, 10),
        format="%Y-%m-%d",
        errors="coerce"
    ).dt.normalize()

    out["email_norm"] = (
        out["email"]
        .map(normalize_email)
    )

    # ad_id llega como ag:120...
    out["ad_id_form"] = (
        out["ad_id"]
        .astype(str)
        .str.replace(
            r"^[A-Za-z]+:",
            "",
            regex=True
        )
        .map(normalize_id)
    )

    out["ad_name"] = (
        out["ad_name"]
        .astype(str)
        .str.strip()
    )

    out["fuente_formulario"] = source_name

    out["estado_normalizado"] = (
        out["state"]
        .map(normalize_mexico_state)
    )

    for question in QUESTION_COLS:
        out[question] = (
            out[question]
            .map(clean_answer)
        )

    # La sección Formularios es SIEMPRE fija:
    # 10 Jul 2026 - 10 Ago 2026.
    out = (
        out[
            out["fecha_formulario"]
            .between(
                FORM_START,
                FORM_END,
                inclusive="both"
            )
        ]
        .copy()
    )

    return out


form_209_video = prepare_form_df(
    form_209_video,
    "209 video"
)

form_209_estatico = prepare_form_df(
    form_209_estatico,
    "209 estático"
)

form_lt_v1 = prepare_form_df(
    form_lt_v1,
    "LT_V1"
)


forms_all_raw = pd.concat(
    [
        form_209_estatico,
        form_209_video,
        form_lt_v1
    ],
    ignore_index=True,
    sort=False
)

# Guardamos todas las filas reales para métricas.
forms_all_raw = forms_all_raw.reset_index(drop=True)


# ============================================================
# 12. FECHA DE ACTUALIZACIÓN
# ============================================================

latest_dates = pd.concat(
    [
        meta["fecha"],
        DT_Leads_fb["fecha"],
        lista_registros["fecha"]
    ],
    ignore_index=True
).dropna()

if not latest_dates.empty:

    latest_date = (
        latest_dates
        .max()
        .strftime("%d.%m.%Y")
    )

    st.caption(
        f"Meta + Zoho · actualizado al {latest_date}"
    )

else:

    st.caption(
        "Meta + Zoho"
    )


# ============================================================
# 13. FILTROS MAESTROS
# ============================================================

all_dates = pd.concat(
    [
        meta["fecha"],
        DT_Leads_fb["fecha"],
        lista_registros["fecha"]
    ],
    ignore_index=True
).dropna()

if all_dates.empty:

    st.error(
        "No existen fechas válidas para construir los filtros."
    )

    st.stop()


min_date = (
    all_dates
    .min()
    .date()
)

max_date = (
    all_dates
    .max()
    .date()
)


# Para el filtro visible usamos los nombres oficiales de Meta.
all_ad_names = sorted(
    set(
        meta[
            "ad_name"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    |
    set(
        forms_all_raw[
            "ad_name"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
)


company_sizes = sorted(
    DT_Leads_fb[
        "Tamaño de la empresa"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .replace("", np.nan)
    .dropna()
    .unique()
)


with st.sidebar:

    st.header(
        "Filtros maestros"
    )

    selected_dates = st.date_input(
        "Fecha",
        value=(
            min_date,
            max_date
        ),
        min_value=min_date,
        max_value=max_date
    )

    if (
        isinstance(
            selected_dates,
            (tuple, list)
        )
        and len(selected_dates) == 2
    ):
        start_date = selected_dates[0]
        end_date = selected_dates[1]

    else:
        start_date = selected_dates
        end_date = selected_dates


    selected_sizes = st.multiselect(
        "Zoho · Tamaño de la empresa",
        options=company_sizes,
        default=[]
    )


    selected_ads = st.multiselect(
        "Anuncio",
        options=all_ad_names,
        default=[],
        help=(
            "Se muestra el nombre del anuncio, "
            "pero internamente Meta y Zoho se filtran mediante el ID."
        )
    )


start_ts = pd.Timestamp(
    start_date
).normalize()

end_ts = pd.Timestamp(
    end_date
).normalize()


# ============================================================
# 14. APLICACIÓN CORRECTA DE FILTROS MAESTROS
# ============================================================

# Primero fecha.
meta_f = (
    meta[
        meta["fecha"].notna()
        &
        meta["fecha"].between(
            start_ts,
            end_ts,
            inclusive="both"
        )
    ]
    .copy()
)

leads_f = (
    DT_Leads_fb[
        DT_Leads_fb["fecha"].between(
            start_ts,
            end_ts,
            inclusive="both"
        )
    ]
    .copy()
)

lista_f = (
    lista_registros[
        lista_registros["fecha"].between(
            start_ts,
            end_ts,
            inclusive="both"
        )
    ]
    .copy()
)


# Después anuncio.
#
# La selección visible es por NOMBRE,
# pero el filtro real se convierte a IDs oficiales de Meta.
if selected_ads:

    selected_ad_ids = set()

    for ad_name in selected_ads:
        selected_ad_ids.update(
            name_to_ids.get(
                ad_name,
                []
            )
        )

    meta_f = (
        meta_f[
            meta_f["ad_id"].isin(
                selected_ad_ids
            )
        ]
        .copy()
    )

    leads_f = (
        leads_f[
            leads_f["ad_id"].isin(
                selected_ad_ids
            )
        ]
        .copy()
    )

    lista_f = (
        lista_f[
            lista_f["ad_id"].isin(
                selected_ad_ids
            )
        ]
        .copy()
    )


# Diagnóstico de fechas Meta.
# Permite verificar rápidamente qué fechas entraron realmente al filtro.
with st.sidebar.expander("Diagnóstico fecha Meta"):
    st.write(
        "Rango maestro:",
        start_ts.strftime("%d/%m/%Y"),
        "→",
        end_ts.strftime("%d/%m/%Y")
    )

    st.write(
        "Fecha mínima Meta:",
        meta["fecha"].min().strftime("%d/%m/%Y")
        if meta["fecha"].notna().any()
        else "Sin fecha"
    )

    st.write(
        "Fecha máxima Meta:",
        meta["fecha"].max().strftime("%d/%m/%Y")
        if meta["fecha"].notna().any()
        else "Sin fecha"
    )

    st.write(
        "Filas Meta dentro del filtro:",
        len(meta_f)
    )

    st.write(
        "Fechas Meta inválidas:",
        int(meta["fecha"].isna().sum())
    )

    if not meta_f.empty:
        resumen_mes_meta = (
            meta_f.assign(
                mes=meta_f["fecha"].dt.to_period("M").astype(str)
            )
            .groupby("mes")
            .size()
            .rename("filas")
            .reset_index()
        )

        st.dataframe(
            resumen_mes_meta,
            hide_index=True,
            use_container_width=True
        )


# Tamaño de empresa existe únicamente en Zoho.
if selected_sizes:

    leads_f = (
        leads_f[
            leads_f[
                "Tamaño de la empresa"
            ]
            .astype(str)
            .str.strip()
            .isin(selected_sizes)
        ]
        .copy()
    )


# ============================================================
# 15. META: COMPILADO CORRECTO POR DÍA
# ============================================================

def aggregate_meta_daily_by_ad(df):
    """
    PRIMER NIVEL:
    suma todos los desgloses de plataforma / ubicación / dispositivo
    para cada DÍA + ANUNCIO.

    Así cada anuncio tiene una sola fila por día.
    """

    return (
        df.groupby(
            [
                "fecha",
                "ad_id",
                "ad_name"
            ],
            as_index=False,
            dropna=False
        )
        .agg(
            gasto=(
                "Importe gastado (MXN)",
                "sum"
            ),
            registros=(
                "Resultados",
                "sum"
            ),
            alcance=(
                "Alcance",
                "sum"
            ),
            impresiones=(
                "Impresiones",
                "sum"
            ),
            clicks=(
                "Clics únicos en el enlace",
                "sum"
            )
        )
        .sort_values(
            [
                "fecha",
                "ad_name"
            ]
        )
    )


def aggregate_meta_daily(df):
    """
    SEGUNDO NIVEL:
    suma los anuncios ya compilados para obtener una fila total por día.
    """

    daily_by_ad = aggregate_meta_daily_by_ad(
        df
    )

    daily = (
        daily_by_ad.groupby(
            "fecha",
            as_index=False
        )
        .agg(
            gasto=("gasto", "sum"),
            registros=("registros", "sum"),
            alcance=("alcance", "sum"),
            impresiones=("impresiones", "sum"),
            clicks=("clicks", "sum")
        )
        .sort_values(
            "fecha"
        )
    )

    daily["ctr"] = np.where(
        daily["alcance"] > 0,
        daily["clicks"]
        / daily["alcance"]
        * 100,
        np.nan
    )

    daily["cpr"] = np.where(
        daily["registros"] > 0,
        daily["gasto"]
        / daily["registros"],
        np.nan
    )

    daily[
        "conversion_click_registro"
    ] = np.where(
        daily["clicks"] > 0,
        daily["registros"]
        / daily["clicks"]
        * 100,
        np.nan
    )

    return daily


def aggregate_meta_ad(df):
    """
    Compila primero por día + anuncio y después suma todo el periodo.
    """

    daily_by_ad = aggregate_meta_daily_by_ad(
        df
    )

    return (
        daily_by_ad.groupby(
            [
                "ad_id",
                "ad_name"
            ],
            as_index=False,
            dropna=False
        )
        .agg(
            gasto=("gasto", "sum"),
            registros=("registros", "sum"),
            alcance=("alcance", "sum"),
            impresiones=("impresiones", "sum"),
            clicks=("clicks", "sum")
        )
        .sort_values(
            "ad_name"
        )
    )


meta_daily_ad = (
    aggregate_meta_daily_by_ad(
        meta_f
    )
)

meta_daily = (
    aggregate_meta_daily(
        meta_f
    )
)

meta_by_ad = (
    aggregate_meta_ad(
        meta_f
    )
)


# ============================================================
# 16. REGLAS CENTRALES DEL CUSTOMER JOURNEY
# ============================================================

# SQL se define exclusivamente por Estado de Posible cliente.
SQL_HANDSHAKE = "E4 - SQL transferido - Handshake"

# Etapas comerciales que cuentan como "En proceso de venta" DESPUÉS
# de haber pasado por SQL_HANDSHAKE.
SALES_PROCESS_STAGES = {
    norm_text(x)
    for x in [
        "En espera de pago",
        "Análisis de Necesidades",
        "Calificación",
        "Propuesta/Cotización",
        "Prueba",
        "Negociación/Revisión",
        "Pagado - Cierre pero en proceso de validación por posible fraude",
    ]
}

CLOSE_STAGE = "Contrato Firmado - Cierre Logrado"

# ------------------------------------------------------------
# Estados usados en Perfilamiento-SQL
# ------------------------------------------------------------
PROFILE_DISCARD_STATES = {
    norm_text(x)
    for x in [
        "E2 - Descartado - Imposible contactar",
        "E3 - Descarte - No apto por producto o comercial",
        "E3 - Descarte falla criterio SQL",
        "E1 - Descartado - No procede",
        "Descartado - Proyecto - No urge (solicita info y costos)",
        "Descartado - No tienen la necesidad",
        "E2 - Descartado - Sin apertura a conversar",
        "Descartado - Propuesta pan básico enviada, 3 intentos sin respuesta y sin información de la empresa",
        "Descartado - La necesidad era otra",
        "E3 - Descarte - Necesidad fuera de la oferta",
        "Descartado - Curioseando (pidió info)",
        "Descartado - Luego del primer contacto no dio apertura a videollamada ni cotización",
        "Descartado - Duplicado",
        "Descartado - Llamada interna, Chat, Mensaje, DEMO - Soporte/Ventas",
        "Descartado - No cumple mínimo viable para Perfilamiento",
        "Descartado - Otro",
        "Descartado - Quería utilizar su número móvil como principal del conmutador",
        "Descartado - Prueba Cp (Interna)",
        "Descartado - Es una notificación de lead de pago",
        "Descartado - Info incorrecta",
        "Descartado - Fraude",
        "Descartado - Lead no apto para ventas (AV/Presupuesto/Celulares)",
        "Descartado - Ya es cliente",
        "Descartado - Nos contactará más adelante (no tiene la urgencia)",
        "Descartado - Lo requieren para más adelante",
        "Descartado - Imposible contactar",
        "Descartado por Growth",
        "GR - Descartado por Growth",
        "E2 - Descartado - Rechazo sin apertura",
        "Descartado - Benchmark",
        "Descartado - Ya lo tienen resuelto",
        "Descartado - No decide_Sin urgencia_Sin prioridad_Sin fecha de implementación",
        "Descartado - Fuera de su presupuesto / Se le hizo caro",
        "Descartado - Se intentó recontactar pero no responde por ningún medio",
        "Descartado - No le interesa",
        "Descartado - Se decidió por otro servicio",
        "Descartado - Interesado en un sistema propio y no en un esquema de renta mensual",
        "Descartado - Se queda con su proveedor actual",
        "E0 - No es lead",
        "FP - No es lead",
        "Retomar en el futuro - Nos contactará más adelante (no tiene la urgencia)",
    ]
}

PROFILE_IN_PROCESS_STATES = {
    norm_text(x)
    for x in [
        "E2 - Intento de contacto iniciado",
        "E3 - En proceso - Agenda pendiente",
        "Nurturing / Lead from DEMO",
        "E3 - Cliente existente (Upsale o Soporte)",
        "E3 - Nurturing",
        "Contactado",
        "Reactivado por Perfilamiento",
        "Intento de contacto iniciado",
    ]
}

PROFILE_TRANSFER_STATES = {
    norm_text("E4-Transferencia Fallida"),
    norm_text("E4 - Transferencia Fallida"),
    norm_text("E4 - Handshake no consumado"),
}

PROFILE_NOT_STARTED_STATES = {
    norm_text("Intento de contacto no iniciado"),
    norm_text("E4 - Intento de contacto no iniciado"),
}


def lead_stages(df):
    """
    Reglas únicas del funnel. Estas reglas alimentan Lead Journey,
    Formularios, heatmaps y bubble plots.

    Registros -> SQL -> En proceso de venta -> Cierre
    """
    registros = df.copy()

    estado_norm = registros["Estado de Posible cliente"].map(norm_text)
    stage_norm = registros["Stage"].map(norm_text)

    # SQL = únicamente Handshake.
    sql = registros[estado_norm == norm_text(SQL_HANDSHAKE)].copy()

    # En proceso de venta = SQL + Stage dentro del catálogo comercial.
    sql_stage_norm = sql["Stage"].map(norm_text)
    en_proceso_venta = sql[sql_stage_norm.isin(SALES_PROCESS_STAGES)].copy()

    # Cierre = SQL + contrato firmado. Es una etapa independiente del catálogo
    # anterior porque el Stage de cierre no debe mezclarse con proceso activo.
    cierre = sql[sql_stage_norm == norm_text(CLOSE_STAGE)].copy()

    # Bases auxiliares para las gráficas de pastel.
    perfilamiento = registros[estado_norm != norm_text(SQL_HANDSHAKE)].copy()
    ventas_fuera_proceso = sql[
        ~sql_stage_norm.isin(SALES_PROCESS_STAGES)
        & (sql_stage_norm != norm_text(CLOSE_STAGE))
    ].copy()

    return {
        "Registros": registros,
        "Perfilamiento": perfilamiento,
        "SQL": sql,
        "En proceso de venta": en_proceso_venta,
        "Ventas": ventas_fuera_proceso,
        "Cierre": cierre,
    }


def profiling_groups(df):
    """
    Grupos operativos exclusivos de la sección Perfilamiento-SQL.

    IMPORTANTE:
    - SQL NO es una subcategoría de Perfilamiento: es la etapa del funnel definida
      exactamente igual que en Lead Journey.
    - Los demás elementos son subgrupos operativos de Perfilamiento y no deben
      reutilizarse como etapas del funnel.
    """
    out = df.copy()
    estado_norm = out["Estado de Posible cliente"].map(norm_text)

    sql = out[estado_norm == norm_text(SQL_HANDSHAKE)].copy()
    descartes = out[estado_norm.isin(PROFILE_DISCARD_STATES)].copy()
    en_proceso = out[estado_norm.isin(PROFILE_IN_PROCESS_STATES)].copy()
    en_transferencia = out[estado_norm.isin(PROFILE_TRANSFER_STATES)].copy()
    contacto_no_iniciado = out[estado_norm.isin(PROFILE_NOT_STARTED_STATES)].copy()

    return {
        "SQL": sql,
        "En proceso de perfilamiento": en_proceso,
        "En transferencia": en_transferencia,
        "Descartes perfilamiento": descartes,
        "Contacto no iniciado": contacto_no_iniciado,
    }


journey = lead_stages(leads_f)
profile_journey = profiling_groups(leads_f)

# ============================================================
# 17. BASE FIJA PARA LA SECCIÓN FORMULARIOS
# ============================================================

# Esta base NO usa el filtro maestro de fecha.
# Siempre trabaja del 10 Jul 2026 al 10 Ago 2026.

DT_Leads_fb_form_window = (
    DT_Leads_fb[
        DT_Leads_fb["fecha"]
        .between(
            FORM_START,
            FORM_END,
            inclusive="both"
        )
    ]
    .copy()
)

# Funnel completo de la ventana fija.
form_window_journey = lead_stages(
    DT_Leads_fb_form_window
)


def email_set(df):
    return set(
        df["email_norm"]
        .dropna()
        .astype(str)
    )


emails_zoho_window = email_set(
    DT_Leads_fb_form_window
)

emails_sql = email_set(
    form_window_journey["SQL"]
)

emails_process = email_set(
    form_window_journey["En proceso de venta"]
)

emails_close = email_set(
    form_window_journey["Cierre"]
)


# Una fila representativa de Zoho por correo para añadir
# tamaño de empresa y campos descriptivos.
zoho_window_by_email = (
    DT_Leads_fb_form_window[
        DT_Leads_fb_form_window["email_norm"].notna()
    ]
    .sort_values("fecha")
    .drop_duplicates(subset=["email_norm"], keep="last")
    [
        [
            "email_norm",
            "Tamaño de la empresa",
            "Nombre completo",
            "Estatus de lead",
            "Estado de Posible cliente",
            "Fuente de Posible cliente",
            "utm_content",
            "Hora de creación",
            "Fecha de calificación",
            "Stage",
            "Importe",
        ]
    ]
    .copy()
)

zoho_window_by_email = zoho_window_by_email.rename(
    columns={
        "Tamaño de la empresa": "tamano_empresa_zoho",
        "Nombre completo": "nombre_zoho",
        "Estatus de lead": "estatus_lead_zoho",
        "Estado de Posible cliente": "estado_posible_cliente_zoho",
        "Fuente de Posible cliente": "fuente_zoho",
        "utm_content": "utm_content_zoho",
        "Hora de creación": "hora_creacion_zoho",
        "Fecha de calificación": "fecha_calificacion_zoho",
        "Stage": "stage_zoho",
        "Importe": "importe_zoho",
    }
)

forms_enriched = forms_all_raw.merge(
    zoho_window_by_email,
    on="email_norm",
    how="left"
)

forms_enriched["en_zoho"] = forms_enriched["email_norm"].astype(str).isin(emails_zoho_window)
forms_enriched["llego_sql"] = forms_enriched["email_norm"].astype(str).isin(emails_sql)
forms_enriched["llego_proceso"] = forms_enriched["email_norm"].astype(str).isin(emails_process)
forms_enriched["llego_cierre"] = forms_enriched["email_norm"].astype(str).isin(emails_close)


def classify_form_phase(row):
    """
    Etapas visuales de Formularios tomadas EXCLUSIVAMENTE del funnel principal.
    Los grupos operativos de Perfilamiento-SQL no intervienen aquí.
    """
    if row["llego_cierre"]:
        return "Cierre"
    if row["llego_proceso"]:
        return "En proceso de venta"
    if row["llego_sql"]:
        return "SQL"
    return "Lead bruto"


forms_enriched["fase_formulario"] = forms_enriched.apply(classify_form_phase, axis=1)

# ============================================================
# 18. LEADS REGISTRADOS ANTERIORMENTE
# ============================================================

# 1) lista_registros dentro de la misma ventana fija.
lista_form_window = (
    lista_registros[
        lista_registros["fecha"]
        .between(
            FORM_START,
            FORM_END,
            inclusive="both"
        )
    ]
    .copy()
)

# 2) Quitar los registros que SÍ aparecen en Zoho
# dentro de la ventana + fuente Facebook.
lista_no_zoho_window = (
    lista_form_window[
        ~lista_form_window[
            "email_norm"
        ]
        .astype(str)
        .isin(
            emails_zoho_window
        )
    ]
    .copy()
)

# 3) Buscar esos correos en TODO DT_Leads,
# sin filtro de fecha ni Fuente de Posible cliente.
dt_full_history = (
    DT_Leads[
        DT_Leads[
            "email_norm"
        ].notna()
    ]
    .sort_values(
        "fecha"
    )
    .drop_duplicates(
        subset=["email_norm"],
        keep="last"
    )
    .copy()
)

historical_keep = [
    "email_norm",
    "Nombre completo",
    "Correo electrónico",
    "Estatus de lead",
    "Fuente de Posible cliente",
    "utm_content",
    "Hora de creación",
    "Fecha de calificación",
    "Tamaño de la empresa"
]

historical_found = (
    lista_no_zoho_window[
        [
            "email_norm"
        ]
    ]
    .drop_duplicates()
    .merge(
        dt_full_history[
            historical_keep
        ],
        on="email_norm",
        how="inner"
    )
)

# 4) Añadir las preguntas del formulario.
# Si hay más de un envío del mismo correo,
# usamos el envío más reciente.
forms_latest_email = (
    forms_all_raw[
        forms_all_raw[
            "email_norm"
        ].notna()
    ]
    .sort_values(
        "fecha_formulario"
    )
    .drop_duplicates(
        subset=["email_norm"],
        keep="last"
    )
    [
        [
            "email_norm",
            "ad_name",
            "fecha_formulario",
            "fuente_formulario"
        ]
        + QUESTION_COLS
    ]
    .copy()
)

leads_registrados_anteriormente_base = (
    historical_found
    .merge(
        forms_latest_email,
        on="email_norm",
        how="left"
    )
)


# ============================================================
# 19. FILTROS DE FORMULARIOS
# ============================================================

# NO se usa selected_dates.
forms_f = forms_enriched.copy()
historical_f = leads_registrados_anteriormente_base.copy()

# Filtro maestro por anuncio: directo por nombre.
if selected_ads:

    forms_f = (
        forms_f[
            forms_f[
                "ad_name"
            ]
            .isin(
                selected_ads
            )
        ]
        .copy()
    )

    historical_f = (
        historical_f[
            historical_f[
                "ad_name"
            ]
            .isin(
                selected_ads
            )
        ]
        .copy()
    )

# Filtro maestro por tamaño de empresa.
if selected_sizes:

    forms_f = (
        forms_f[
            forms_f[
                "tamano_empresa_zoho"
            ]
            .astype(str)
            .str.strip()
            .isin(
                selected_sizes
            )
        ]
        .copy()
    )

    historical_f = (
        historical_f[
            historical_f[
                "Tamaño de la empresa"
            ]
            .astype(str)
            .str.strip()
            .isin(
                selected_sizes
            )
        ]
        .copy()
    )


# Para análisis respuesta -> avance usamos UN LEAD por correo + anuncio.
# Las métricas de formularios conservan las filas reales.
forms_analysis_base = (
    forms_f[
        forms_f[
            "email_norm"
        ].notna()
    ]
    .sort_values(
        "fecha_formulario"
    )
    .drop_duplicates(
        subset=[
            "email_norm",
            "ad_name"
        ],
        keep="last"
    )
    .copy()
)


# ============================================================
# 20. TABLA ANALÍTICA DE TRANSICIONES
# ============================================================

TRANSITIONS = [
    {
        "name": "Lead bruto → SQL",
        "origin": None,
        "target": "llego_sql"
    },
    {
        "name": "SQL → En proceso de venta",
        "origin": "llego_sql",
        "target": "llego_proceso"
    },
    {
        "name": "En proceso de venta → Cierre",
        "origin": "llego_proceso",
        "target": "llego_cierre"
    }
]


def build_transition_table(df):
    """
    Unidad:
    anuncio × transición × pregunta × respuesta
    """

    rows = []

    if df.empty:
        return pd.DataFrame(
            columns=[
                "anuncio",
                "transición",
                "pregunta",
                "pregunta_corta",
                "respuesta",
                "n_origen",
                "n_avanzaron",
                "prob_avance",
                "prob_general_anuncio",
                "lift"
            ]
        )

    for ad_name, ad_df in df.groupby(
        "ad_name",
        dropna=False
    ):

        for transition in TRANSITIONS:

            if transition["origin"] is None:
                origin_df = ad_df.copy()
            else:
                origin_df = (
                    ad_df[
                        ad_df[
                            transition["origin"]
                        ]
                    ]
                    .copy()
                )

            if origin_df.empty:
                continue

            n_general_origin = len(
                origin_df
            )

            n_general_advanced = int(
                origin_df[
                    transition["target"]
                ]
                .sum()
            )

            prob_general = safe_div(
                n_general_advanced,
                n_general_origin
            )

            for question in QUESTION_COLS:

                q_df = origin_df.copy()

                q_df[question] = (
                    q_df[question]
                    .fillna(
                        "Sin respuesta"
                    )
                    .astype(str)
                    .replace(
                        "",
                        "Sin respuesta"
                    )
                )

                grouped = (
                    q_df.groupby(
                        question,
                        dropna=False
                    )
                )

                for response, response_df in grouped:

                    n_origin = len(
                        response_df
                    )

                    n_advanced = int(
                        response_df[
                            transition["target"]
                        ]
                        .sum()
                    )

                    prob_advance = safe_div(
                        n_advanced,
                        n_origin
                    )

                    lift = (
                        safe_div(
                            prob_advance,
                            prob_general
                        )
                        if (
                            pd.notna(
                                prob_general
                            )
                            and prob_general > 0
                        )
                        else np.nan
                    )

                    rows.append(
                        {
                            "anuncio": ad_name,
                            "transición": transition["name"],
                            "pregunta": question,
                            "pregunta_corta": QUESTION_LABELS.get(
                                question,
                                question
                            ),
                            "respuesta": response,
                            "n_origen": n_origin,
                            "n_avanzaron": n_advanced,
                            "prob_avance": prob_advance,
                            "prob_general_anuncio": prob_general,
                            "lift": lift
                        }
                    )

    return pd.DataFrame(
        rows
    )


tabla_transiciones = build_transition_table(
    forms_analysis_base
)


# ============================================================
# 21. NAVEGACIÓN
# ============================================================

tab_meta, tab_journey, tab_profile, tab_forms = st.tabs(
    [
        "Meta vista general",
        "Lead Journey",
        "Perfilamiento-SQL",
        "Formularios"
    ]
)


# ============================================================
# ============================================================
# PÁGINA 1: META VISTA GENERAL
# ============================================================
# ============================================================

with tab_meta:

    st.subheader(
        "Métricas generales"
    )


    # IMPORTANTE:
    # Todas las métricas se toman del compilado diario,
    # NO de las filas crudas de los desgloses.
    total_spend = (
        meta_daily[
            "gasto"
        ]
        .sum()
    )

    total_results = (
        meta_daily[
            "registros"
        ]
        .sum()
    )

    total_reach = (
        meta_daily[
            "alcance"
        ]
        .sum()
    )

    total_clicks = (
        meta_daily[
            "clicks"
        ]
        .sum()
    )


    overall_ctr = safe_div(
        total_clicks,
        total_reach,
        100
    )

    overall_cpr = safe_div(
        total_spend,
        total_results
    )

    overall_conv = safe_div(
        total_results,
        total_clicks,
        100
    )


    zoho_registros = len(
        lista_f
    )

    diferencia_registros = (
        total_results
        -
        zoho_registros
    )


    n_days = (
        meta_daily[
            "fecha"
        ]
        .nunique()
    )

    avg_daily_spend = safe_div(
        total_spend,
        n_days
    )


    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "CTR",
        pct(
            overall_ctr
        )
    )

    c2.metric(
        "Costo por Registro (CPR)",
        money(
            overall_cpr
        )
    )

    c3.metric(
        "Conversión clic → registro",
        pct(
            overall_conv
        )
    )

    c4.metric(
        "Registros Meta",
        integer(
            total_results
        )
    )


    c5, c6, c7 = st.columns(
        3
    )

    c5.metric(
        "Diferencia registros",
        integer(
            diferencia_registros
        )
    )

    c6.metric(
        "Importe gastado",
        money(
            total_spend
        )
    )

    c7.metric(
        "Gasto diario promedio",
        money(
            avg_daily_spend
        )
    )


    st.caption(
        "Las filas de Meta se compilan primero por día + anuncio, "
        "sumando todos los desgloses de plataforma, ubicación y dispositivo. "
        "Después se calculan CTR, CPR y conversión sobre esos totales."
    )


    # ========================================================
    # TABLA DE CONTROL DEL COMPILADO
    # ========================================================

    with st.expander(
        "Ver compilado diario de Meta"
    ):

        st.dataframe(
            meta_daily_ad,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # FUNNEL META
    # ========================================================

    st.divider()

    st.subheader(
        "Funnel de Meta por anuncio"
    )

    if meta_by_ad.empty:

        st.info(
            "No hay datos para los filtros seleccionados."
        )

    else:

        fig_funnel = go.Figure()

        for _, row in meta_by_ad.iterrows():

            fig_funnel.add_trace(
                go.Funnel(
                    name=row["ad_name"],
                    y=[
                        "Alcance",
                        "Clics únicos en el enlace",
                        "Registros"
                    ],
                    x=[
                        row["alcance"],
                        row["clicks"],
                        row["registros"]
                    ],
                    textinfo=(
                        "value+percent initial"
                    )
                )
            )

        fig_funnel.update_layout(
            height=520,
            legend_title_text="Anuncio",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig_funnel,
            use_container_width=True
        )


    # ========================================================
    # SERIES DE TIEMPO
    # ========================================================

    st.divider()

    st.subheader(
        "Series de tiempo"
    )


    left, right = st.columns(
        2
    )


    with left:

        st.markdown(
            "**Importe gastado vs. registros**"
        )

        fig_ts_1 = make_subplots(
            specs=[
                [
                    {
                        "secondary_y": True
                    }
                ]
            ]
        )

        fig_ts_1.add_trace(
            go.Scatter(
                x=meta_daily["fecha"],
                y=meta_daily["gasto"],
                name="Importe gastado",
                mode="lines+markers"
            ),
            secondary_y=False
        )

        fig_ts_1.add_trace(
            go.Scatter(
                x=meta_daily["fecha"],
                y=meta_daily["registros"],
                name="Registros",
                mode="lines+markers"
            ),
            secondary_y=True
        )

        fig_ts_1.update_yaxes(
            title_text="MXN",
            secondary_y=False
        )

        fig_ts_1.update_yaxes(
            title_text="Registros",
            secondary_y=True
        )

        fig_ts_1.update_xaxes(
            title_text="Día"
        )

        fig_ts_1.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig_ts_1,
            use_container_width=True
        )


    with right:

        st.markdown(
            "**Costo por Registro vs. CTR**"
        )

        fig_ts_2 = make_subplots(
            specs=[
                [
                    {
                        "secondary_y": True
                    }
                ]
            ]
        )

        fig_ts_2.add_trace(
            go.Scatter(
                x=meta_daily["fecha"],
                y=meta_daily["cpr"],
                name="Costo por Registro",
                mode="lines+markers"
            ),
            secondary_y=False
        )

        fig_ts_2.add_trace(
            go.Scatter(
                x=meta_daily["fecha"],
                y=meta_daily["ctr"],
                name="CTR",
                mode="lines+markers"
            ),
            secondary_y=True
        )

        fig_ts_2.update_yaxes(
            title_text="CPR (MXN)",
            secondary_y=False
        )

        fig_ts_2.update_yaxes(
            title_text="CTR (%)",
            secondary_y=True
        )

        fig_ts_2.update_xaxes(
            title_text="Día"
        )

        fig_ts_2.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )

        st.plotly_chart(
            fig_ts_2,
            use_container_width=True
        )


    # ========================================================
    # SANKEY
    # ========================================================

    st.divider()

    st.subheader(
        "Plataforma y dispositivos"
    )

    st.caption(
        "Flujo: Registros → Plataforma → Ubicación"
    )


    sankey_base = (
        meta_f.groupby(
            [
                "Plataforma",
                "Ubicación"
            ],
            dropna=False,
            as_index=False
        )
        .agg(
            registros=(
                "Resultados",
                "sum"
            )
        )
    )

    sankey_base = (
        sankey_base[
            sankey_base[
                "registros"
            ]
            > 0
        ]
        .copy()
    )


    if sankey_base.empty:

        st.info(
            "No hay registros suficientes para construir el Sankey."
        )

    else:

        sankey_base[
            "Plataforma"
        ] = (
            sankey_base[
                "Plataforma"
            ]
            .fillna(
                "Sin plataforma"
            )
        )

        sankey_base[
            "Ubicación"
        ] = (
            sankey_base[
                "Ubicación"
            ]
            .fillna(
                "Sin ubicación"
            )
        )

        platforms = list(
            sankey_base[
                "Plataforma"
            ]
            .astype(str)
            .unique()
        )

        sankey_base[
            "location_node"
        ] = (
            sankey_base[
                "Plataforma"
            ]
            .astype(str)
            + " · "
            + sankey_base[
                "Ubicación"
            ]
            .astype(str)
        )

        location_nodes = list(
            sankey_base[
                "location_node"
            ]
            .unique()
        )

        node_keys = (
            ["Registros"]
            + platforms
            + location_nodes
        )

        labels = (
            ["Registros"]
            + platforms
            + [
                x.split(
                    " · ",
                    1
                )[1]
                if " · " in x
                else x
                for x
                in location_nodes
            ]
        )

        idx = {
            key: i
            for i, key
            in enumerate(
                node_keys
            )
        }

        source = []
        target = []
        value = []

        platform_totals = (
            sankey_base.groupby(
                "Plataforma",
                as_index=False
            )[
                "registros"
            ]
            .sum()
        )

        for _, row in platform_totals.iterrows():

            source.append(
                idx["Registros"]
            )

            target.append(
                idx[
                    row[
                        "Plataforma"
                    ]
                ]
            )

            value.append(
                row[
                    "registros"
                ]
            )

        for _, row in sankey_base.iterrows():

            source.append(
                idx[
                    row[
                        "Plataforma"
                    ]
                ]
            )

            target.append(
                idx[
                    row[
                        "location_node"
                    ]
                ]
            )

            value.append(
                row[
                    "registros"
                ]
            )

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=18,
                        thickness=18,
                        label=labels
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value
                    )
                )
            ]
        )

        fig_sankey.update_layout(
            height=600,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )

        st.plotly_chart(
            fig_sankey,
            use_container_width=True
        )


# ============================================================
# ============================================================
# PÁGINA 2: LEAD JOURNEY
# ============================================================
# ============================================================

with tab_journey:

    st.subheader("Customer Journey")

    registros_n = len(journey["Registros"])
    sql_n = len(journey["SQL"])
    proceso_n = len(journey["En proceso de venta"])
    cierre_n = len(journey["Cierre"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", integer(registros_n))
    m2.metric("SQL", integer(sql_n))
    m3.metric("En proceso de venta", integer(proceso_n))
    m4.metric("Cierre", integer(cierre_n))

    st.caption(
        'Base: Fuente de Posible cliente ∈ {"Facebook Ads", "FacebookAds"}. '
        "MQL se elimina de este dashboard alterno."
    )

    # ========================================================
    # FUNNEL PRINCIPAL + TASAS DE CONVERSIÓN GENERALES
    # ========================================================
    st.divider()
    st.subheader("Funnel de Lead Journey")

    funnel_order = ["Registros", "SQL", "En proceso de venta", "Cierre"]
    funnel_counts = {
        "Registros": registros_n,
        "SQL": sql_n,
        "En proceso de venta": proceso_n,
        "Cierre": cierre_n,
    }

    # Conversión simple = etapa actual / etapa inmediatamente anterior.
    # Conversión acumulada = etapa actual / Registros.
    conversion_rows = []
    previous_stage = None

    for stage in funnel_order:
        current_n = funnel_counts[stage]

        if previous_stage is None:
            simple_rate = 100.0 if current_n > 0 else np.nan
        else:
            previous_n = funnel_counts[previous_stage]
            simple_rate = safe_div(current_n, previous_n, mult=100)

        accumulated_rate = safe_div(
            current_n,
            registros_n,
            mult=100
        )

        conversion_rows.append(
            {
                "Etapa": stage,
                "Leads": current_n,
                "Conversión simple": simple_rate,
                "Conversión acumulada": accumulated_rate,
            }
        )

        previous_stage = stage

    conversion_table = pd.DataFrame(conversion_rows)

    # Texto del funnel general: muestra la conversión SIMPLE en cada etapa.
    funnel_text = []
    for _, row in conversion_table.iterrows():
        simple_txt = (
            f"{row['Conversión simple']:.2f}%"
            if pd.notna(row["Conversión simple"])
            else "—"
        )
        funnel_text.append(
            f"{int(row['Leads']):,}<br>{simple_txt} vs etapa anterior"
        )

    col_funnel, col_rules = st.columns([2.2, 1])

    with col_funnel:
        fig_general_funnel = go.Figure(
            go.Funnel(
                y=funnel_order,
                x=[funnel_counts[s] for s in funnel_order],
                text=funnel_text,
                textinfo="text",
                customdata=np.column_stack(
                    [
                        conversion_table["Conversión simple"].to_numpy(),
                        conversion_table["Conversión acumulada"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "Etapa: %{y}<br>"
                    "Leads: %{x}<br>"
                    "Conversión simple: %{customdata[0]:.2f}%<br>"
                    "Conversión acumulada: %{customdata[1]:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig_general_funnel.update_layout(
            height=560,
            margin=dict(l=20, r=90, t=20, b=20)
        )

        st.plotly_chart(
            fig_general_funnel,
            use_container_width=True
        )

    with col_rules:
        st.markdown("### Reglas generales")
        st.markdown(
            f"""
- **Registros:** todos los registros de `DT_Leads` después de filtros maestros.
- **SQL:** `Estado de Posible cliente = {SQL_HANDSHAKE}`.
- **En proceso de venta:** primero cumple SQL y luego `Stage` pertenece al catálogo comercial definido.
- **Cierre:** primero cumple SQL y luego `Stage = {CLOSE_STAGE}`.
- **Conversión simple:** etapa actual ÷ etapa anterior.
- **Conversión acumulada:** etapa actual ÷ Registros.
- **MQL:** eliminado de este dashboard alterno.
            """
        )

    st.markdown("### Tasas de conversión generales")

    conversion_display = conversion_table.copy()
    conversion_display["Conversión simple"] = conversion_display[
        "Conversión simple"
    ].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
    conversion_display["Conversión acumulada"] = conversion_display[
        "Conversión acumulada"
    ].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

    st.dataframe(
        conversion_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Etapa": st.column_config.TextColumn("Etapa"),
            "Leads": st.column_config.NumberColumn("Leads", format="%d"),
            "Conversión simple": st.column_config.TextColumn(
                "% simple (vs etapa anterior)"
            ),
            "Conversión acumulada": st.column_config.TextColumn(
                "% acumulado (desde Registros)"
            ),
        },
    )

    # El cierre se define por el Stage actual, mientras "En proceso de venta"
    # contiene únicamente stages comerciales activos. Si un lead ya cerró,
    # deja de estar en "En proceso de venta"; por eso el cociente simple
    # proceso -> cierre debe interpretarse como relación entre stocks actuales,
    # no como una cohorte histórica de transición.
    if proceso_n > 0 and cierre_n > proceso_n:
        st.warning(
            "El número de cierres supera a los leads actualmente 'En proceso de venta'. "
            "Esto puede ocurrir porque Cierre y En proceso se clasifican por Stage actual. "
            "Para una tasa histórica estricta En proceso → Cierre se necesita historial "
            "de cambios de etapa por lead."
        )

    st.markdown("### Funnel por anuncio")
    st.caption(
        "Este gráfico conserva el desglose por anuncio. "
        "La tabla y el funnel anteriores son los totales generales con los filtros activos."
    )

    ads_journey = sorted(
        leads_f["ad_name"].dropna().astype(str).unique()
    )
    fig_lead_funnel = go.Figure()

    for ad in ads_journey:
        values = []
        hover_details = []

        for stage in funnel_order:
            stage_df = journey[stage][
                journey[stage]["ad_name"] == ad
            ].copy()
            values.append(len(stage_df))

            if stage == "Cierre" and not stage_df.empty:
                detail_lines = []
                for _, r in stage_df.iterrows():
                    lead_id = (
                        str(r.get(LEAD_ID_COL, "—"))
                        if LEAD_ID_COL is not None
                        else "—"
                    )
                    importe = money(r.get("importe_num", np.nan))
                    detail_lines.append(
                        f"ID: {lead_id} · Importe: {importe}"
                    )
                hover_details.append("<br>".join(detail_lines))
            else:
                hover_details.append("")

        fig_lead_funnel.add_trace(
            go.Funnel(
                name=ad,
                y=funnel_order,
                x=values,
                customdata=np.array(hover_details, dtype=object),
                textinfo="value+percent initial",
                hovertemplate=(
                    "Anuncio: %{fullData.name}<br>"
                    "Etapa: %{y}<br>"
                    "Leads: %{x}<br>"
                    "%{customdata}"
                    "<extra></extra>"
                )
            )
        )

    fig_lead_funnel.update_layout(
        height=620,
        legend_title_text="Anuncio",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(
        fig_lead_funnel,
        use_container_width=True
    )

    # ========================================================
    # PASTELES: PERFILAMIENTO Y VENTAS
    # ========================================================
    st.divider()
    st.subheader("Perfilamiento y Ventas")

    def show_pie(title, data, reason_col):
        if data.empty:
            st.info(f"{title}: sin datos para los filtros actuales.")
            return
        reasons = (
            data[reason_col]
            .fillna("Sin motivo especificado")
            .astype(str).str.strip().replace("", "Sin motivo especificado")
            .value_counts().reset_index()
        )
        reasons.columns = ["Motivo", "Leads"]
        fig = px.pie(reasons, names="Motivo", values="Leads", hole=0.48, title=title)
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    p1, p2 = st.columns(2)
    with p1:
        show_pie(
            "Perfilamiento",
            journey["Perfilamiento"],
            "Estado de Posible cliente"
        )
    with p2:
        show_pie(
            "Ventas",
            journey["Ventas"],
            "Stage"
        )

    # ========================================================
    # HEATMAP ACUMULADO
    # ========================================================
    st.divider()
    st.subheader("Probabilidad simple acumulada por anuncio")
    st.caption(
        "Color = probabilidad acumulada desde Registros. "
        "Número = cantidad de leads en la etapa."
    )

    heat_stages = ["Registros", "SQL", "En proceso de venta", "Cierre"]
    ads_heat = sorted(leads_f["ad_name"].dropna().astype(str).unique())
    z_prob, text_count = [], []

    for stage in heat_stages:
        probs_row, counts_row = [], []
        for ad in ads_heat:
            base_n = len(journey["Registros"][journey["Registros"]["ad_name"] == ad])
            stage_n = len(journey[stage][journey[stage]["ad_name"] == ad])
            probs_row.append(stage_n / base_n * 100 if base_n > 0 else np.nan)
            counts_row.append(stage_n)
        z_prob.append(probs_row)
        text_count.append(counts_row)

    if ads_heat:
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=z_prob,
                x=ads_heat,
                y=heat_stages,
                text=text_count,
                texttemplate="%{text}",
                hovertemplate=(
                    "Anuncio: %{x}<br>Etapa: %{y}<br>Leads: %{text}<br>"
                    "Probabilidad acumulada: %{z:.2f}%<extra></extra>"
                ),
                zmin=0, zmax=100,
                colorscale=[
                    [0.00, "#ffffff"], [0.20, "#fee5d9"],
                    [0.40, "#fcae91"], [0.60, "#fb6a4a"],
                    [0.80, "#de2d26"], [1.00, "#a50f15"]
                ],
                colorbar=dict(title="Probabilidad %")
            )
        )
        fig_heat.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=heat_stages
        )
        fig_heat.update_layout(
            height=420,
            xaxis_title="Anuncio",
            yaxis_title="Etapa",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No hay anuncios disponibles para el heatmap.")

    # ========================================================
    # TABLAS POR ETAPA + TOTAL
    # ========================================================
    st.divider()
    st.subheader("Datos filtrados por etapa del funnel")

    for stage in ["SQL", "En proceso de venta", "Cierre"]:
        st.markdown(f"### {stage}")
        st.dataframe(
            journey[stage].sort_values("fecha", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Todos los datos")
    st.dataframe(
        leads_f.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PÁGINA 2B: PERFILAMIENTO-SQL
# ============================================================

with tab_profile:
    st.subheader("Perfilamiento-SQL")
    st.markdown("### Descartes Perfilamiento")
    st.caption(
        "Esta sección usa DT_Leads y responde a fecha, tamaño de empresa y anuncio "
        "de los filtros maestros. El filtro de anuncio de esta sección es un refuerzo adicional."
    )

    # Métricas generales: responden únicamente a los filtros maestros.
    metric_order = [
        "SQL",
        "En proceso de perfilamiento",
        "En transferencia",
        "Descartes perfilamiento",
        "Contacto no iniciado",
    ]
    profile_groups_master = profiling_groups(leads_f)
    cols = st.columns(5)
    for col, group_name in zip(cols, metric_order):
        col.metric(group_name, integer(len(profile_groups_master[group_name])))

    # Refuerzo por anuncio: se coloca debajo de las métricas generales y
    # afecta las visualizaciones y tablas de esta sección, no las métricas anteriores.
    profile_ad_options = sorted(leads_f["ad_name"].dropna().astype(str).unique())
    selected_profile_ads = st.multiselect(
        "Refuerzo por anuncio",
        options=profile_ad_options,
        default=[],
        key="profile_ad_filter",
        help="Filtro adicional exclusivo de Perfilamiento-SQL; se aplica después de los filtros maestros."
    )

    profile_base = leads_f.copy()
    if selected_profile_ads:
        profile_base = profile_base[profile_base["ad_name"].isin(selected_profile_ads)].copy()

    profile_groups = profiling_groups(profile_base)

    # Pasteles por grupo.
    st.divider()
    st.subheader("Distribución por grupo")

    pie_groups = [
        ("SQL", profile_groups["SQL"]),
        ("En proceso de perfilamiento", profile_groups["En proceso de perfilamiento"]),
        ("En transferencia", profile_groups["En transferencia"]),
        ("Descartes perfilamiento", profile_groups["Descartes perfilamiento"]),
        ("Contacto no iniciado", profile_groups["Contacto no iniciado"]),
    ]

    for i in range(0, len(pie_groups), 2):
        c1, c2 = st.columns(2)
        for target_col, item in zip([c1, c2], pie_groups[i:i+2]):
            title, data = item
            with target_col:
                show_pie(title, data, "Estado de Posible cliente")

    # Series de tiempo.
    st.divider()
    st.subheader("Evolución diaria")
    ts1, ts2 = st.columns(2)

    with ts1:
        daily_sql = (
            profile_groups["SQL"].groupby("fecha").size()
            .rename("SQL").reset_index()
        )
        fig = px.line(
            daily_sql, x="fecha", y="SQL", markers=True,
            title="SQL por día"
        )
        fig.update_layout(height=420, xaxis_title="Día", yaxis_title="Leads SQL")
        st.plotly_chart(fig, use_container_width=True)

    with ts2:
        daily_discard = (
            profile_groups["Descartes perfilamiento"].groupby("fecha").size()
            .rename("Leads descartados").reset_index()
        )
        fig = px.line(
            daily_discard, x="fecha", y="Leads descartados", markers=True,
            title="Leads descartados por día"
        )
        fig.update_layout(height=420, xaxis_title="Día", yaxis_title="Leads")
        st.plotly_chart(fig, use_container_width=True)

    # Toques: media e IC95% por grupo.
    st.divider()
    st.subheader("Toques de Perfilamiento")
    st.caption("Media de Conteo e intervalo de confianza aproximado del 95% por grupo.")

    touch_rows = []
    for group_name in metric_order:
        vals = profile_groups[group_name]["conteo_num"].dropna().astype(float)
        n = len(vals)
        mean = vals.mean() if n else np.nan
        sd = vals.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        ci_low = mean - 1.96 * se if n > 1 else np.nan
        ci_high = mean + 1.96 * se if n > 1 else np.nan
        touch_rows.append({
            "Grupo": group_name,
            "N": n,
            "Media": mean,
            "CI_low": ci_low,
            "CI_high": ci_high,
        })

    touch_summary = pd.DataFrame(touch_rows)
    valid_touch = touch_summary[touch_summary["N"] > 0].copy()

    if valid_touch.empty:
        st.info("No hay valores numéricos disponibles en la columna Conteo.")
    else:
        valid_touch["err_plus"] = valid_touch["CI_high"] - valid_touch["Media"]
        valid_touch["err_minus"] = valid_touch["Media"] - valid_touch["CI_low"]
        fig_touch = go.Figure(
            go.Bar(
                x=valid_touch["Grupo"],
                y=valid_touch["Media"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=valid_touch["err_plus"],
                    arrayminus=valid_touch["err_minus"],
                    visible=True,
                ),
                customdata=valid_touch[["N", "CI_low", "CI_high"]].to_numpy(),
                hovertemplate=(
                    "Grupo: %{x}<br>Media: %{y:.2f}<br>N: %{customdata[0]}<br>"
                    "IC95%: %{customdata[1]:.2f} – %{customdata[2]:.2f}<extra></extra>"
                )
            )
        )
        fig_touch.update_layout(
            height=470,
            xaxis_title="Grupo",
            yaxis_title="Media de toques",
            margin=dict(l=20, r=20, t=30, b=100)
        )
        st.plotly_chart(fig_touch, use_container_width=True)
        st.dataframe(touch_summary, use_container_width=True, hide_index=True)

    # Tablas por grupo.
    st.divider()
    st.subheader("Datos por grupo de Perfilamiento-SQL")
    for group_name in metric_order:
        st.markdown(f"### {group_name}")
        st.dataframe(
            profile_groups[group_name].sort_values("fecha", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# ============================================================
# PÁGINA 3: FORMULARIOS
# ============================================================
# ============================================================

with tab_forms:

    st.subheader(
        "Formularios"
    )

    st.info(
        "Nota: se consideran los formularios de los anuncios "
        "209 estático, 209 video y LT_V1. "
        "LT_V3 tuvo únicamente un formulario y se descartó. "
        "Para los anuncios 209 se tomó únicamente el formulario "
        "más reciente actualizado y homologable."
    )

    st.caption(
        "Periodo fijo de esta sección: 10 Jul 2026 – 10 Ago 2026. "
        "El filtro maestro de fecha NO modifica esta pestaña. "
        "Sí se aplican los filtros maestros de anuncio y tamaño de empresa."
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    st.subheader(
        "Métricas generales"
    )

    formularios_meta_n = len(
        forms_f
    )

    formularios_zoho_n = int(
        forms_f[
            "en_zoho"
        ]
        .sum()
    )

    formularios_209_estatico_n = int(
        (
            (
                forms_f[
                    "fuente_formulario"
                ]
                ==
                "209 estático"
            )
            &
            forms_f[
                "en_zoho"
            ]
        ).sum()
    )

    formularios_209_video_n = int(
        (
            (
                forms_f[
                    "fuente_formulario"
                ]
                ==
                "209 video"
            )
            &
            forms_f[
                "en_zoho"
            ]
        ).sum()
    )

    formularios_lt_v1_n = int(
        (
            (
                forms_f[
                    "fuente_formulario"
                ]
                ==
                "LT_V1"
            )
            &
            forms_f[
                "en_zoho"
            ]
        ).sum()
    )

    registros_anteriores_n = len(
        historical_f
    )


    k1, k2, k3 = st.columns(
        3
    )

    k1.metric(
        "Formularios Meta",
        integer(
            formularios_meta_n
        )
    )

    k2.metric(
        "Formularios en Zoho",
        integer(
            formularios_zoho_n
        )
    )

    k3.metric(
        "Formularios 209 estático",
        integer(
            formularios_209_estatico_n
        )
    )


    k4, k5, k6 = st.columns(
        3
    )

    k4.metric(
        "Formularios 209 video",
        integer(
            formularios_209_video_n
        )
    )

    k5.metric(
        "Formularios LT_V1",
        integer(
            formularios_lt_v1_n
        )
    )

    k6.metric(
        "Registros anteriores",
        integer(
            registros_anteriores_n
        )
    )

    st.caption(
        "Formularios Meta cuenta los envíos reales de los tres CSV. "
        "Los conteos por anuncio consideran únicamente formularios cuyo correo "
        "sí fue localizado en Zoho dentro de la ventana fija y con Fuente de Posible cliente Facebook."
    )



    # ========================================================
    # MAPA DE REGISTROS POR ESTADO
    # ========================================================

    st.divider()

    st.subheader(
        "Distribución de formularios en Zoho por estado"
    )

    st.caption(
        "Solo se muestran formularios cuyo correo fue encontrado en Zoho. "
        "El mapa no responde al filtro maestro de fecha; sí responde a anuncio "
        "y tamaño de empresa. Valores de estado no reconocibles se excluyen."
    )

    map_base = (
        forms_f[
            forms_f[
                "en_zoho"
            ]
            &
            forms_f[
                "estado_normalizado"
            ]
            .notna()
        ]
        .copy()
    )

    state_counts = (
        map_base.groupby(
            "estado_normalizado",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "registros"
            }
        )
    )

    if state_counts.empty:

        st.info(
            "No hay estados válidos para construir el mapa con los filtros actuales."
        )

    else:

        state_counts[
            "geojson_state_name"
        ] = (
            state_counts[
                "estado_normalizado"
            ]
            .map(
                MEXICO_STATE_TO_GEOJSON
            )
        )

        state_counts = (
            state_counts[
                state_counts[
                    "geojson_state_name"
                ]
                .notna()
            ]
            .copy()
        )

        try:

            mexico_geojson = (
                cargar_geojson_mexico()
            )

            map_col, table_col = st.columns(
                [
                    2.2,
                    1
                ]
            )

            with map_col:

                fig_map = px.choropleth(
                    state_counts,
                    geojson=mexico_geojson,
                    locations="geojson_state_name",
                    featureidkey="properties.state_name",
                    color="registros",
                    hover_name="estado_normalizado",
                    hover_data={
                        "registros": True,
                        "geojson_state_name": False
                    },
                    color_continuous_scale="Reds",
                    labels={
                        "registros": "Registros"
                    },
                    title="Registros por estado"
                )

                fig_map.update_geos(
                    fitbounds="locations",
                    visible=False
                )

                fig_map.update_layout(
                    height=560,
                    margin=dict(
                        l=0,
                        r=0,
                        t=50,
                        b=0
                    ),
                    coloraxis_colorbar=dict(
                        title="Registros"
                    )
                )

                st.plotly_chart(
                    fig_map,
                    use_container_width=True
                )

            with table_col:

                st.markdown(
                    "**Frecuencia por estado**"
                )

                state_table = (
                    state_counts[
                        [
                            "estado_normalizado",
                            "registros"
                        ]
                    ]
                    .rename(
                        columns={
                            "estado_normalizado": "Estado",
                            "registros": "Registros"
                        }
                    )
                    .sort_values(
                        "Registros",
                        ascending=False
                    )
                )

                st.dataframe(
                    state_table,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.warning(
                "No fue posible cargar el mapa geográfico de México. "
                "Se muestra la tabla de frecuencia por estado."
            )

            state_table = (
                state_counts[
                    [
                        "estado_normalizado",
                        "registros"
                    ]
                ]
                .rename(
                    columns={
                        "estado_normalizado": "Estado",
                        "registros": "Registros"
                    }
                )
                .sort_values(
                    "Registros",
                    ascending=False
                )
            )

            st.dataframe(
                state_table,
                use_container_width=True,
                hide_index=True
            )

        invalid_state_n = int(
            (
                forms_f[
                    "en_zoho"
                ]
                &
                forms_f[
                    "estado_normalizado"
                ]
                .isna()
            )
            .sum()
        )

        if invalid_state_n > 0:

            st.caption(
                f"{invalid_state_n} formularios localizados en Zoho "
                "tenían un valor de estado no reconocible y no se incluyeron en el mapa."
            )


    # ========================================================
    # FRECUENCIAS POR PREGUNTA Y FASE
    # ========================================================

    st.divider()

    st.subheader(
        "Respuestas de formulario por etapa del funnel"
    )

    st.caption(
        "La fase visual usa únicamente las etapas del funnel principal: Lead bruto, SQL, "
        "En proceso de venta y Cierre. Los subgrupos de Perfilamiento-SQL no se usan como etapas."
    )


    phase_order = [
        "Lead bruto",
        "SQL",
        "En proceso de venta",
        "Cierre"
    ]

    frequency_df = forms_analysis_base.copy()

    for i, question in enumerate(
        QUESTION_COLS
    ):

        question_label = QUESTION_LABELS.get(
            question,
            question
        )

        freq = (
            frequency_df.groupby(
                [
                    question,
                    "fase_formulario"
                ],
                dropna=False
            )
            .size()
            .rename(
                "n"
            )
            .reset_index()
        )

        freq = (
            freq.rename(
                columns={
                    question: "respuesta"
                }
            )
        )

        freq[
            "fase_formulario"
        ] = pd.Categorical(
            freq[
                "fase_formulario"
            ],
            categories=phase_order,
            ordered=True
        )

        fig_freq = px.bar(
            freq,
            x="respuesta",
            y="n",
            color="fase_formulario",
            barmode="stack",
            title=question_label,
            labels={
                "respuesta": "Respuesta",
                "n": "Leads",
                "fase_formulario": "Fase"
            }
        )

        fig_freq.update_layout(
            height=430,
            xaxis_tickangle=-30,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=120
            )
        )

        st.plotly_chart(
            fig_freq,
            use_container_width=True
        )


    # ========================================================
    # ANÁLISIS DE PROBABILIDAD SIMPLE Y LIFT
    # ========================================================

    st.divider()

    st.subheader(
        "Asociación entre respuestas y avance del funnel"
    )

    st.markdown(
        """
La unidad analítica es **Anuncio × Transición × Pregunta × Respuesta**.

La probabilidad se calcula secuencialmente:

- Lead bruto → SQL: todos los leads del anuncio con esa respuesta son el denominador.
- SQL → En proceso de venta: solo los leads que ya llegaron a SQL (`E4 - SQL transferido - Handshake`).
- En proceso de venta → Cierre: solo los leads que ya llegaron a proceso de venta.

`lift = probabilidad de avance de la respuesta / probabilidad general del anuncio`.

Un lift mayor a 1 indica una tasa de avance superior al promedio de ese anuncio y transición.
        """
    )


    transition_options = [
        x["name"]
        for x in TRANSITIONS
    ]

    question_options = [
        QUESTION_LABELS[
            q
        ]
        for q in QUESTION_COLS
    ]


    f1, f2, f3 = st.columns(
        3
    )

    with f1:

        selected_transitions = st.multiselect(
            "Transición",
            options=transition_options,
            default=transition_options,
            key="forms_transition_filter"
        )

    with f2:

        selected_questions_labels = st.multiselect(
            "Pregunta",
            options=question_options,
            default=question_options,
            key="forms_question_filter"
        )

    with f3:

        min_n = st.number_input(
            "N mínimo para visualizaciones",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
            help=(
                "Por defecto solo se interpretan respuestas con n_origen >= 20. "
                "El N se calcula después de segmentar por anuncio, transición, "
                "pregunta y respuesta."
            ),
            key="forms_min_n"
        )


    selected_questions_raw = [
        raw
        for raw, label
        in QUESTION_LABELS.items()
        if label
        in selected_questions_labels
    ]


    analytical_f = (
        tabla_transiciones[
            tabla_transiciones[
                "transición"
            ]
            .isin(
                selected_transitions
            )
            &
            tabla_transiciones[
                "pregunta"
            ]
            .isin(
                selected_questions_raw
            )
        ]
        .copy()
    )

    # Clasificación de muestra.
    analytical_f[
        "estado_muestra"
    ] = np.where(
        analytical_f[
            "n_origen"
        ]
        >= min_n,
        "Elegible para interpretación",
        "Muestra insuficiente para interpretación"
    )

    # SOLO esta base alimenta heatmaps y bubble plots.
    # El dataset detallado conserva todas las respuestas.
    analytical_vis = (
        analytical_f[
            analytical_f[
                "n_origen"
            ]
            >= min_n
        ]
        .copy()
    )


    # ========================================================
    # HEATMAP POR ANUNCIO
    # ========================================================

    st.markdown(
        "### Heatmap de lift por anuncio"
    )

    st.caption(
        f"Solo se muestran combinaciones con n_origen >= {min_n}. "
        "Color = lift. Texto = probabilidad de avance y N del origen. "
        "La comparación siempre se realiza dentro del mismo anuncio."
    )


    analytical_ads = sorted(
        analytical_vis[
            "anuncio"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    if not analytical_ads:

        st.info(
            "No hay datos suficientes para construir el análisis con los filtros actuales."
        )

    else:

        for ad_name in analytical_ads:

            ad_table = (
                analytical_vis[
                    analytical_vis[
                        "anuncio"
                    ]
                    == ad_name
                ]
                .copy()
            )

            if ad_table.empty:
                continue

            ad_table[
                "pregunta_respuesta"
            ] = (
                ad_table[
                    "pregunta_corta"
                ]
                .astype(str)
                + " — "
                + ad_table[
                    "respuesta"
                ]
                .astype(str)
            )

            # Pivot para heatmap.
            heat_lift = (
                ad_table.pivot_table(
                    index="pregunta_respuesta",
                    columns="transición",
                    values="lift",
                    aggfunc="first"
                )
                .reindex(
                    columns=[
                        x
                        for x in transition_options
                        if x in selected_transitions
                    ]
                )
            )

            heat_prob = (
                ad_table.pivot_table(
                    index="pregunta_respuesta",
                    columns="transición",
                    values="prob_avance",
                    aggfunc="first"
                )
                .reindex(
                    index=heat_lift.index,
                    columns=heat_lift.columns
                )
            )

            heat_n = (
                ad_table.pivot_table(
                    index="pregunta_respuesta",
                    columns="transición",
                    values="n_origen",
                    aggfunc="first"
                )
                .reindex(
                    index=heat_lift.index,
                    columns=heat_lift.columns
                )
            )

            text_matrix = []

            for row_idx in heat_lift.index:

                text_row = []

                for col_name in heat_lift.columns:

                    p = heat_prob.loc[
                        row_idx,
                        col_name
                    ]

                    n = heat_n.loc[
                        row_idx,
                        col_name
                    ]

                    if pd.isna(p):
                        text_row.append(
                            ""
                        )
                    else:
                        text_row.append(
                            f"{p * 100:.0f}% | n={int(n)}"
                        )

                text_matrix.append(
                    text_row
                )


            # Customdata: n_origen, n_avanzaron, prob, promedio, lift, pregunta, respuesta
            custom_matrix = []

            lookup = (
                ad_table.set_index(
                    [
                        "pregunta_respuesta",
                        "transición"
                    ]
                )
            )

            for row_idx in heat_lift.index:

                custom_row = []

                for col_name in heat_lift.columns:

                    key = (
                        row_idx,
                        col_name
                    )

                    if key in lookup.index:

                        row = lookup.loc[
                            key
                        ]

                        if isinstance(
                            row,
                            pd.DataFrame
                        ):
                            row = row.iloc[0]

                        custom_row.append(
                            [
                                row[
                                    "pregunta_corta"
                                ],
                                row[
                                    "respuesta"
                                ],
                                row[
                                    "n_origen"
                                ],
                                row[
                                    "n_avanzaron"
                                ],
                                row[
                                    "prob_avance"
                                ],
                                row[
                                    "prob_general_anuncio"
                                ],
                                row[
                                    "lift"
                                ]
                            ]
                        )

                    else:

                        custom_row.append(
                            [
                                "",
                                "",
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan
                            ]
                        )

                custom_matrix.append(
                    custom_row
                )


            # Escala centrada en 1:
            # menor que promedio -> claro
            # mayor que promedio -> más intenso.
            max_lift = (
                np.nanmax(
                    heat_lift.values
                )
                if np.isfinite(
                    heat_lift.values
                ).any()
                else 2
            )

            zmax = max(
                2.0,
                float(
                    max_lift
                )
            )

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=heat_lift.values,
                    x=list(
                        heat_lift.columns
                    ),
                    y=list(
                        heat_lift.index
                    ),
                    text=text_matrix,
                    texttemplate="%{text}",
                    customdata=np.array(
                        custom_matrix,
                        dtype=object
                    ),
                    zmin=0,
                    zmax=zmax,
                    colorscale=[
                        [0.00, "#f7fbff"],
                        [0.25, "#deebf7"],
                        [0.50, "#9ecae1"],
                        [0.75, "#4292c6"],
                        [1.00, "#084594"]
                    ],
                    colorbar=dict(
                        title="Lift"
                    ),
                    hovertemplate=(
                        f"Anuncio: {ad_name}<br>"
                        "Pregunta: %{customdata[0]}<br>"
                        "Respuesta: %{customdata[1]}<br>"
                        "Transición: %{x}<br>"
                        "N en etapa: %{customdata[2]}<br>"
                        "Avanzaron: %{customdata[3]}<br>"
                        "Probabilidad: %{customdata[4]:.1%}<br>"
                        "Promedio anuncio: %{customdata[5]:.1%}<br>"
                        "Lift: %{customdata[6]:.2f}×"
                        "<extra></extra>"
                    )
                )
            )

            fig_heat.update_layout(
                title=ad_name,
                height=max(
                    520,
                    28
                    * len(
                        heat_lift.index
                    )
                ),
                xaxis_title="Transición",
                yaxis_title="Pregunta — Respuesta",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig_heat,
                use_container_width=True
            )


    # ========================================================
    # BUBBLE PLOT POR ANUNCIO
    # ========================================================

    st.markdown(
        "### Probabilidad de avance por transición"
    )

    st.caption(
        f"Solo se muestran combinaciones con n_origen >= {min_n}. "
        "X = transición, Y = probabilidad de avance, tamaño de burbuja = N. "
        "Cada burbuja representa una combinación pregunta–respuesta."
    )


    for ad_name in analytical_ads:

        bubble = (
            analytical_vis[
                analytical_vis[
                    "anuncio"
                ]
                == ad_name
            ]
            .copy()
        )

        if bubble.empty:
            continue

        bubble[
            "pregunta_respuesta"
        ] = (
            bubble[
                "pregunta_corta"
            ]
            .astype(str)
            + " — "
            + bubble[
                "respuesta"
            ]
            .astype(str)
        )

        bubble[
            "prob_avance_pct"
        ] = (
            bubble[
                "prob_avance"
            ]
            * 100
        )

        bubble[
            "lift_text"
        ] = (
            bubble[
                "lift"
            ]
            .map(
                lambda x:
                f"{x:.2f}×"
                if pd.notna(x)
                else "—"
            )
        )

        fig_bubble = px.scatter(
            bubble,
            x="transición",
            y="prob_avance_pct",
            size="n_origen",
            color="pregunta_corta",
            hover_name="pregunta_respuesta",
            hover_data={
                "n_origen": True,
                "n_avanzaron": True,
                "prob_avance_pct": ":.1f",
                "prob_general_anuncio": ":.1%",
                "lift_text": True,
                "pregunta_corta": False
            },
            size_max=55,
            title=ad_name,
            labels={
                "transición": "Transición",
                "prob_avance_pct": "Probabilidad de avance (%)",
                "pregunta_corta": "Pregunta",
                "n_origen": "N origen",
                "n_avanzaron": "Avanzaron",
                "prob_general_anuncio": "Promedio anuncio",
                "lift_text": "Lift"
            }
        )

        fig_bubble.update_layout(
            height=560,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig_bubble,
            use_container_width=True
        )


    # ========================================================
    # TABLA FINAL DE LIFT
    # ========================================================

    st.divider()

    st.subheader(
        "Tabla analítica de avance y lift"
    )

    table_display = analytical_f.copy()

    if not table_display.empty:

        table_display[
            "prob_avance"
        ] = (
            table_display[
                "prob_avance"
            ]
            * 100
        )

        table_display[
            "prob_general_anuncio"
        ] = (
            table_display[
                "prob_general_anuncio"
            ]
            * 100
        )

        table_display = (
            table_display[
                [
                    "anuncio",
                    "transición",
                    "pregunta_corta",
                    "respuesta",
                    "n_origen",
                    "n_avanzaron",
                    "prob_avance",
                    "prob_general_anuncio",
                    "lift",
                    "estado_muestra"
                ]
            ]
            .rename(
                columns={
                    "pregunta_corta": "pregunta",
                    "prob_avance": "prob_avance_%",
                    "prob_general_anuncio": "prob_general_anuncio_%"
                }
            )
            .sort_values(
                [
                    "anuncio",
                    "transición",
                    "pregunta",
                    "lift"
                ],
                ascending=[
                    True,
                    True,
                    True,
                    False
                ]
            )
        )

    st.caption(
        "La tabla conserva también respuestas con N menor al mínimo seleccionado. "
        "Esas filas aparecen como “Muestra insuficiente para interpretación” y "
        "no alimentan los heatmaps ni bubble plots."
    )

    st.dataframe(
        table_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # LEADS REGISTRADOS ANTERIORMENTE
    # ========================================================

    st.divider()

    st.subheader(
        "Leads registrados anteriormente"
    )

    st.caption(
        "Registros del periodo 10 Jul–10 Ago que no aparecieron como nuevos leads "
        "en Zoho con fuente Facebook durante esa ventana, pero cuyo correo sí fue "
        "localizado en DT_Leads histórico."
    )


    historical_display_cols = [
        "Nombre completo",
        "Correo electrónico",
        "Estado de Posible cliente",
        "Fuente de Posible cliente",
        "utm_content",
        "Hora de creación",
        "Fecha de calificación"
    ] + QUESTION_COLS


    historical_display_cols = [
        col
        for col in historical_display_cols
        if col in historical_f.columns
    ]


    st.dataframe(
        historical_f[
            historical_display_cols
        ],
        use_container_width=True,
        hide_index=True
    )
