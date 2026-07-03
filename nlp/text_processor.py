import re


_REGIONES = [
    "arica", "parinacota", "tarapacá", "tarapaca", "antofagasta",
    "atacama", "coquimbo", "valparaíso", "valparaiso", "metropolitana",
    "o'higgins", "ohiggins", "maule", "ñuble", "nuble", "biobío", "biobio",
    "araucanía", "araucania", "los ríos", "los rios", "los lagos",
    "aysén", "aysen", "magallanes",
]

_COMUNAS = [
    "santiago", "providencia", "las condes", "vitacura", "ñuñoa", "nunoa",
    "maipú", "maipu", "pudahuel", "quilicura", "renca", "cerro navia",
    "lo prado", "estación central", "estacion central", "la florida",
    "puente alto", "san bernardo", "buin", "paine", "colina", "lampa",
    "til til", "melipilla", "talagante", "peñaflor", "penalfor",
    "valparaíso", "valparaiso", "viña del mar", "vina del mar",
    "concepción", "concepcion", "talcahuano", "temuco", "rancagua",
    "iquique", "antofagasta", "la serena", "coquimbo", "puerto montt",
    "osorno", "valdivia", "punta arenas", "copiapó", "copiapo",
    "curicó", "curico", "talca", "chillán", "chillan", "los ángeles",
    "los angeles",
]

_KEYWORDS = ["región", "region", "comuna", "ciudad", "provincia", "sector",
             "km", "kilómetro", "kilómetros", "av.", "avenida", "calle",
             "ruta", "camino"]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(
        _REGIONES + _COMUNAS, key=len, reverse=True
    )) + r")\b",
    re.IGNORECASE,
)

_CONTEXT_PATTERN = re.compile(
    r"(?:región|region|comuna|ciudad|provincia|sector|en|de)\s+([\w\s]{2,40}?)(?:\s*,|\s*\.|$)",
    re.IGNORECASE,
)


class TextProcessor:
    def extraer_ubicacion(self, texto: str) -> str:
        if not texto:
            return ""

        matches = _PATTERN.findall(texto)
        if matches:
            seen = []
            for m in matches:
                if m.lower() not in [s.lower() for s in seen]:
                    seen.append(m)
            return ", ".join(seen)

        ctx = _CONTEXT_PATTERN.search(texto)
        if ctx:
            return ctx.group(1).strip()

        return ""
