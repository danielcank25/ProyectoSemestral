"""CLI para indexar data/ con FAISS+SBERT y hacer consultas directas a esa base.

Uso:
    python search.py build
    python search.py query "pregunta en lenguaje natural" [--licitacion 2788-68-LP25] [--top 5]
"""
import argparse
import os
import sys

from nlp.rag_index import RagIndex

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")

DOC_TYPE_LABELS = {
    "bases_tecnicas": "Bases técnicas",
    "bases_administrativas": "Bases administrativas",
    "itemizado": "Itemizado",
}


def cmd_build(_args):
    rag = RagIndex(INDEX_DIR)
    info = rag.build(DATA_DIR)
    print(f"Índice construido: {info['chunks']} fragmentos en {INDEX_DIR}")
    for w in info["warnings"]:
        print(f"  ! {w}")


def _print_hit(hit, indent=""):
    label = DOC_TYPE_LABELS.get(hit["doc_type"], hit["doc_type"])
    loc = f"pág. {hit['page']}" if hit.get("page") else hit.get("item", "")
    print(f"{indent}[{hit['score']:.3f}] ({label} · {hit['licitacion']} · {hit['file']} · {loc})")
    print(f"{indent}  {hit['text'][:300]}")
    for rel in hit.get("related_bases_tecnicas") or []:
        rloc = f"pág. {rel['page']}" if rel.get("page") else ""
        print(f"{indent}    ↳ [{rel['score']:.3f}] {rel['file']} {rloc}: {rel['text'][:200]}")


def cmd_query(args):
    rag = RagIndex(INDEX_DIR)
    if not rag.load():
        print("No hay índice construido. Ejecuta primero: python search.py build", file=sys.stderr)
        sys.exit(1)

    results = rag.search(args.question, top_k=args.top, licitacion=args.licitacion)
    if not results:
        print("Sin resultados.")
        return

    for hit in results:
        _print_hit(hit)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="Indexa los documentos de data/")

    p_query = sub.add_parser("query", help="Consulta el índice")
    p_query.add_argument("question", help="Pregunta en lenguaje natural")
    p_query.add_argument("--top", type=int, default=5, help="Cantidad de resultados")
    p_query.add_argument("--licitacion", default=None, help="Filtrar por licitación, ej. 2788-68-LP25")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
