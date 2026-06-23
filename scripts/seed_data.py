"""
R&D Nexus seed script — MariaDB / Milvus / Neo4j

Usage:
  python scripts/seed_data.py                     # seed all (skips unconfigured stores)
  python scripts/seed_data.py --target milvus
  python scripts/seed_data.py --target neo4j
  python scripts/seed_data.py --target mariadb
  python scripts/seed_data.py --clear             # clear before seeding

  python scripts/seed_data.py \\
      --mariadb-url "mysql+pymysql://rnd:rnd_password@localhost:3306/rnd_nexus" \\
      --milvus-host localhost --milvus-port 19530 \\
      --neo4j-uri bolt://localhost:7687 \\
      --neo4j-username neo4j --neo4j-password password
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import Any
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# sys.path: add mcp-server/src so infrastructure.* imports work
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

MCP_SRC = REPO_ROOT / "mcp-server" / "src"
sys.path.insert(0, str(MCP_SRC))

FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"

FIXTURE_FILES = {
    "papers": FIXTURES_DIR / "papers.json",
    "patents": FIXTURES_DIR / "patents.json",
    "researchers": FIXTURES_DIR / "researchers.json",
    "technologies": FIXTURES_DIR / "technologies.json",
    "projects": FIXTURES_DIR / "projects.json",
}

NODE_TYPE_MAP = {
    "papers": "Paper",
    "patents": "Patent",
    "researchers": "Researcher",
    "technologies": "Technology",
    "projects": "Project",
}


def load_fixtures() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for key, path in FIXTURE_FILES.items():
        with open(path, encoding="utf-8") as f:
            data[key] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# MariaDB
# ---------------------------------------------------------------------------
def seed_mariadb(mariadb_url: str, clear: bool) -> None:
    print("\n[MariaDB] Seeding...")
    try:
        from infrastructure.database import ensure_schema, seed_from_fixtures
    except ImportError as e:
        print(f"[MariaDB] Skipping - missing dependency: {e}")
        return

    try:
        ensure_schema(mariadb_url)
        seed_from_fixtures(mariadb_url, str(FIXTURES_DIR), clear=clear)
        print("[MariaDB] Done.")
    except Exception as e:
        print(f"[MariaDB] Error seeding MariaDB: {e}")


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------
def seed_milvus(
    host: str,
    port: int,
    collection: str,
    clear: bool,
    sentence_model: str,
) -> None:
    print("\n[Milvus] Seeding...")
    try:
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[Milvus] Skipping - missing dependency: {e}")
        return

    from infrastructure.milvus import ensure_collection

    client = MilvusClient(uri="http://" + host + ":" + str(port))

    if clear and client.has_collection(collection):
        client.drop_collection(collection)
        print("[Milvus] Dropped collection '" + collection + "'.")

    ensure_collection(client, collection)

    print("[Milvus] Loading sentence model '" + sentence_model + "' ...")
    model = SentenceTransformer(sentence_model)

    fixtures = load_fixtures()
    rows_inserted = 0

    for fixture_key, entities in fixtures.items():
        node_type = NODE_TYPE_MAP[fixture_key]
        texts = [e.get("text", "") for e in entities]
        entity_ids = [
            e.get("id", e.get(fixture_key[:-1] + "_id", str(i)))
            for i, e in enumerate(entities)
        ]

        if not texts:
            continue

        print("[Milvus]   Embedding " + str(len(texts)) + " " + node_type + " nodes...")
        dense_vectors = model.encode(texts, show_progress_bar=False).tolist()

        data = [
            {
                "id": entity_ids[i],
                "node_type": node_type,
                "name": entities[i].get("title", entities[i].get("name", "")),
                "year": int(entities[i].get("year", 0)),
                "text": texts[i],
                "dense": dense_vectors[i],
            }
            for i in range(len(texts))
        ]
        client.insert(collection_name=collection, data=data)
        rows_inserted += len(data)

    client.flush(collection_name=collection)
    print("[Milvus] Inserted " + str(rows_inserted) + " nodes into '" + collection + "'. Done.")


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
def seed_neo4j(uri: str, username: str, password: str, clear: bool) -> None:
    print("\n[Neo4j] Seeding...")
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        print(f"[Neo4j] Skipping - missing dependency: {e}")
        return

    driver = GraphDatabase.driver(uri, auth=(username, password))
    fixtures = load_fixtures()

    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
            print("[Neo4j] Cleared all nodes and relationships.")

        # Constraints
        for label, prop in [
            ("Paper", "paper_id"),
            ("Patent", "patent_id"),
            ("Researcher", "researcher_id"),
            ("Technology", "tech_id"),
            ("Project", "project_id"),
        ]:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:" + label + ") REQUIRE n." + prop + " IS UNIQUE"
            )

        # Paper nodes
        print("[Neo4j]   Creating Paper nodes...")
        for p in fixtures["papers"]:
            session.run(
                "MERGE (n:Paper {paper_id: $pid}) "
                "SET n.title = $title, n.year = $year, n.citations = $citations, "
                "n.journal = $journal, n.keywords = $keywords",
                pid=p["paper_id"],
                title=p["title"],
                year=p["year"],
                citations=p.get("citations", 0),
                journal=p.get("journal", ""),
                keywords=p.get("keywords", ""),
            )

        # Patent nodes
        print("[Neo4j]   Creating Patent nodes...")
        for p in fixtures["patents"]:
            session.run(
                "MERGE (n:Patent {patent_id: $pid}) "
                "SET n.title = $title, n.applicant = $applicant, n.year = $year, n.country = $country",
                pid=p["patent_id"],
                title=p["title"],
                applicant=p.get("applicant", ""),
                year=p.get("year", 0),
                country=p.get("country", ""),
            )

        # Technology nodes
        print("[Neo4j]   Creating Technology nodes...")
        for t in fixtures["technologies"]:
            session.run(
                "MERGE (n:Technology {tech_id: $tid}) "
                "SET n.name = $name, n.trl = $trl, n.investment_priority = $ip",
                tid=t["tech_id"],
                name=t["name"],
                trl=t.get("trl", 0),
                ip=t.get("investment_priority", ""),
            )

        # Organization nodes (derived from researcher affiliations)
        orgs: set[str] = {r["affiliation"] for r in fixtures["researchers"]}
        for org in orgs:
            session.run("MERGE (o:Organization {name: $name})", name=org)

        # Researcher nodes + WORKS_AT
        print("[Neo4j]   Creating Researcher nodes + WORKS_AT...")
        for r in fixtures["researchers"]:
            session.run(
                "MERGE (n:Researcher {researcher_id: $rid}) "
                "SET n.name = $name, n.affiliation = $aff, n.h_index = $h, n.specialty = $spec",
                rid=r["researcher_id"],
                name=r["name"],
                aff=r["affiliation"],
                h=r.get("h_index", 0),
                spec=r.get("specialty", ""),
            )
            session.run(
                "MATCH (r:Researcher {researcher_id: $rid}), (o:Organization {name: $org}) "
                "MERGE (r)-[:WORKS_AT]->(o)",
                rid=r["researcher_id"],
                org=r["affiliation"],
            )

        # Project nodes
        print("[Neo4j]   Creating Project nodes...")
        for p in fixtures["projects"]:
            session.run(
                "MERGE (n:Project {project_id: $pid}) "
                "SET n.title = $title, n.organization = $org, n.year = $year, n.status = $status",
                pid=p["project_id"],
                title=p["title"],
                org=p.get("organization", ""),
                year=p.get("year", 0),
                status=p.get("status", ""),
            )

        # AUTHORED: Researcher -> Paper
        print("[Neo4j]   Creating AUTHORED relationships...")
        for r in fixtures["researchers"]:
            for pid in r.get("authored_papers", []):
                session.run(
                    "MATCH (r:Researcher {researcher_id: $rid}), (p:Paper {paper_id: $pid}) "
                    "MERGE (r)-[:AUTHORED]->(p)",
                    rid=r["researcher_id"], pid=pid,
                )

        # INVENTED: Researcher -> Patent
        print("[Neo4j]   Creating INVENTED relationships...")
        for r in fixtures["researchers"]:
            for pat_id in r.get("invented_patents", []):
                session.run(
                    "MATCH (r:Researcher {researcher_id: $rid}), (p:Patent {patent_id: $pid}) "
                    "MERGE (r)-[:INVENTED]->(p)",
                    rid=r["researcher_id"], pid=pat_id,
                )

        # RESEARCHES: Researcher -> Technology
        print("[Neo4j]   Creating RESEARCHES relationships...")
        for r in fixtures["researchers"]:
            for tid in r.get("researches_technologies", []):
                session.run(
                    "MATCH (r:Researcher {researcher_id: $rid}), (t:Technology {tech_id: $tid}) "
                    "MERGE (r)-[:RESEARCHES]->(t)",
                    rid=r["researcher_id"], tid=tid,
                )

        # CITES: Paper -> Paper
        print("[Neo4j]   Creating CITES relationships...")
        for p in fixtures["papers"]:
            for cited_id in p.get("cites", []):
                session.run(
                    "MATCH (a:Paper {paper_id: $aid}), (b:Paper {paper_id: $bid}) "
                    "MERGE (a)-[:CITES]->(b)",
                    aid=p["paper_id"], bid=cited_id,
                )

        # EMPLOYS: Project -> Researcher
        print("[Neo4j]   Creating EMPLOYS relationships...")
        for proj in fixtures["projects"]:
            for rid in proj.get("employs_researchers", []):
                session.run(
                    "MATCH (pr:Project {project_id: $pid}), (r:Researcher {researcher_id: $rid}) "
                    "MERGE (pr)-[:EMPLOYS]->(r)",
                    pid=proj["project_id"], rid=rid,
                )

        # USES: Project -> Technology
        print("[Neo4j]   Creating USES relationships...")
        for proj in fixtures["projects"]:
            for tid in proj.get("uses_technologies", []):
                session.run(
                    "MATCH (pr:Project {project_id: $pid}), (t:Technology {tech_id: $tid}) "
                    "MERGE (pr)-[:USES]->(t)",
                    pid=proj["project_id"], tid=tid,
                )

    driver.close()
    print("[Neo4j] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Seed R&D Nexus data stores")
    parser.add_argument(
        "--target",
        choices=["all", "mariadb", "milvus", "neo4j"],
        default="all",
    )
    parser.add_argument("--clear", action="store_true", help="Clear existing data before seeding")

    parser.add_argument("--mariadb-url", default=os.environ.get("MARIADB_URL", ""))
    parser.add_argument("--milvus-host", default=os.environ.get("MILVUS_HOST", "localhost"))
    parser.add_argument("--milvus-port", type=int, default=int(os.environ.get("MILVUS_PORT", "19530")))
    parser.add_argument("--milvus-collection", default=os.environ.get("MILVUS_COLLECTION", "rnd_nodes"))
    parser.add_argument(
        "--sentence-model",
        default=os.environ.get("SENTENCE_TRANSFORMER_MODEL", "snunlp/KR-SBERT-V40K-klueNLI-augSTS"),
    )
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "password"))

    args = parser.parse_args()

    if args.target in ("all", "mariadb"):
        if not args.mariadb_url:
            print("[MariaDB] Skipping - MARIADB_URL not set.")
        else:
            seed_mariadb(args.mariadb_url, args.clear)

    if args.target in ("all", "milvus"):
        if not os.environ.get("MILVUS_HOST") and args.milvus_host == "localhost" and args.target == "all":
            print("[Milvus] Skipping - MILVUS_HOST not set (use --target milvus to force).")
        else:
            seed_milvus(
                host=args.milvus_host,
                port=args.milvus_port,
                collection=args.milvus_collection,
                clear=args.clear,
                sentence_model=args.sentence_model,
            )

    if args.target in ("all", "neo4j"):
        if not os.environ.get("NEO4J_URI") and args.target == "all":
            print("[Neo4j] Skipping - NEO4J_URI not set (use --target neo4j to force).")
        else:
            seed_neo4j(
                uri=args.neo4j_uri,
                username=args.neo4j_username,
                password=args.neo4j_password,
                clear=args.clear,
            )

    print("\nSeeding complete.")


if __name__ == "__main__":
    main()
