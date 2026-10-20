import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


def get_connection(project_root: Path | None = None) -> psycopg.Connection:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    config_path = project_root / ".env.database"

    if not config_path.is_file():
        raise FileNotFoundError(f"Database configuration not found: {config_path}")

    load_dotenv(config_path, override=True)

    return psycopg.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        connect_timeout=10,
    )


def main() -> None:
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_database(),
                current_user,
                current_setting('server_version'),
                inet_server_port()
            """
        )

        database, user, version, port = cursor.fetchone()

        cursor.execute(
            """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name IN ('staging', 'analytics')
            ORDER BY schema_name
            """
        )

        schemas = [row[0] for row in cursor.fetchall()]

    print(f"Database: {database}")
    print(f"Application user: {user}")
    print(f"Server version: {version}")
    print(f"Port: {port}")
    print(f"Schemas: {schemas}")

    if database != "finance360":
        raise ValueError("Connected to an unexpected database")

    if user != "finance360_app":
        raise ValueError("Connected with an unexpected database role")

    if schemas != ["analytics", "staging"]:
        raise ValueError("Required schemas are missing")


if __name__ == "__main__":
    main()