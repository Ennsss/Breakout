"""DuckDB-based storage for player statistics."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class PlayerDatabase:
    """DuckDB database for storing player statistics from multiple sources.

    Maintains separate tables for each data source (FBref, Transfermarkt, Understat)
    and provides a unified view for querying merged data.
    """

    def __init__(self, db_path: str | Path = "data/players.duckdb"):
        """Initialize database connection.

        Args:
            db_path: Path to DuckDB database file. Use ":memory:" for in-memory DB.
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path

        # Create parent directory if needed
        if self.db_path != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(str(self.db_path))
        self._create_tables()
        logger.info(f"Connected to database: {self.db_path}")

    def _create_tables(self) -> None:
        """Create all tables and views if they don't exist."""
        self._create_fbref_table()
        self._create_transfermarkt_table()
        self._create_understat_table()
        self._create_unified_view()

    def _create_fbref_table(self) -> None:
        """Create FBref players table."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fbref_players (
                -- Identity
                player_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                position VARCHAR,
                team VARCHAR,
                nationality VARCHAR,
                age INTEGER,
                birth_year INTEGER,

                -- Playing time
                games INTEGER,
                games_starts INTEGER,
                minutes INTEGER,
                minutes_90s DOUBLE,

                -- Attacking
                goals INTEGER,
                assists INTEGER,
                goals_assists INTEGER,
                goals_pens INTEGER,
                pens_made INTEGER,
                pens_att INTEGER,
                xg DOUBLE,
                npxg DOUBLE,
                xg_assist DOUBLE,
                npxg_xg_assist DOUBLE,

                -- Cards
                cards_yellow INTEGER,
                cards_red INTEGER,

                -- Progressive actions
                progressive_carries INTEGER,
                progressive_passes INTEGER,
                progressive_passes_received INTEGER,

                -- Shooting
                shots INTEGER,
                shots_on_target INTEGER,
                shots_on_target_pct DOUBLE,
                shots_per90 DOUBLE,
                shots_on_target_per90 DOUBLE,
                goals_per_shot DOUBLE,
                goals_per_shot_on_target DOUBLE,
                average_shot_distance DOUBLE,
                shots_free_kicks INTEGER,
                npxg_per_shot DOUBLE,
                xg_net DOUBLE,
                npxg_net DOUBLE,

                -- Passing
                passes_completed INTEGER,
                passes INTEGER,
                passes_pct DOUBLE,
                passes_total_distance INTEGER,
                passes_progressive_distance INTEGER,
                passes_completed_short INTEGER,
                passes_short INTEGER,
                passes_pct_short DOUBLE,
                passes_completed_medium INTEGER,
                passes_medium INTEGER,
                passes_pct_medium DOUBLE,
                passes_completed_long INTEGER,
                passes_long INTEGER,
                passes_pct_long DOUBLE,
                assisted_shots INTEGER,
                passes_into_final_third INTEGER,
                passes_into_penalty_area INTEGER,
                crosses_into_penalty_area INTEGER,

                -- Defense
                tackles INTEGER,
                tackles_won INTEGER,
                tackles_def_3rd INTEGER,
                tackles_mid_3rd INTEGER,
                tackles_att_3rd INTEGER,
                challenge_tackles INTEGER,
                challenges INTEGER,
                challenge_tackles_pct DOUBLE,
                challenges_lost INTEGER,
                blocks INTEGER,
                blocked_shots INTEGER,
                blocked_passes INTEGER,
                interceptions INTEGER,
                tackles_interceptions INTEGER,
                clearances INTEGER,
                errors INTEGER,

                -- Possession
                touches INTEGER,
                touches_def_pen_area INTEGER,
                touches_def_3rd INTEGER,
                touches_mid_3rd INTEGER,
                touches_att_3rd INTEGER,
                touches_att_pen_area INTEGER,
                touches_live_ball INTEGER,
                take_ons INTEGER,
                take_ons_won INTEGER,
                take_ons_won_pct DOUBLE,
                take_ons_tackled INTEGER,
                take_ons_tackled_pct DOUBLE,
                carries INTEGER,
                carries_distance INTEGER,
                carries_progressive_distance INTEGER,
                carries_into_final_third INTEGER,
                carries_into_penalty_area INTEGER,
                miscontrols INTEGER,
                dispossessed INTEGER,
                passes_received INTEGER,

                -- Metadata
                league VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                source VARCHAR DEFAULT 'fbref',
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (player_id, league, season)
            )
        """)

    def _create_transfermarkt_table(self) -> None:
        """Create Transfermarkt players table."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transfermarkt_players (
                player_id VARCHAR NOT NULL,
                player_slug VARCHAR,
                name VARCHAR NOT NULL,
                position VARCHAR,
                team VARCHAR,
                age INTEGER,
                nationality VARCHAR,
                market_value_eur BIGINT,

                league VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                source VARCHAR DEFAULT 'transfermarkt',
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (player_id, league, season)
            )
        """)

    def _create_understat_table(self) -> None:
        """Create Understat players table."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS understat_players (
                player_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                position VARCHAR,
                team VARCHAR,

                games INTEGER,
                minutes INTEGER,
                minutes_90s DOUBLE,

                goals INTEGER,
                assists INTEGER,
                npg INTEGER,
                xg DOUBLE,
                xa DOUBLE,
                npxg DOUBLE,
                xg_chain DOUBLE,
                xg_buildup DOUBLE,

                xg_per90 DOUBLE,
                xa_per90 DOUBLE,
                npxg_per90 DOUBLE,
                goals_per90 DOUBLE,
                assists_per90 DOUBLE,

                shots INTEGER,
                key_passes INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,

                xg_overperformance DOUBLE,
                xa_overperformance DOUBLE,

                league VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                source VARCHAR DEFAULT 'understat',
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (player_id, league, season)
            )
        """)

    def _create_unified_view(self) -> None:
        """Create unified view joining all sources."""
        self.conn.execute("""
            CREATE OR REPLACE VIEW players_unified AS
            SELECT
                COALESCE(f.name, t.name, u.name) AS name,
                COALESCE(f.team, t.team, u.team) AS team,
                COALESCE(f.position, t.position, u.position) AS position,
                COALESCE(f.league, t.league, u.league) AS league,
                COALESCE(f.season, t.season, u.season) AS season,

                -- Demographics (prefer Transfermarkt)
                COALESCE(t.age, f.age) AS age,
                COALESCE(f.nationality, t.nationality) AS nationality,
                t.market_value_eur,

                -- Playing time (prefer FBref)
                COALESCE(f.minutes, u.minutes) AS minutes,
                COALESCE(f.games, u.games) AS games,

                -- Goals & Assists
                COALESCE(f.goals, u.goals) AS goals,
                COALESCE(f.assists, u.assists) AS assists,

                -- xG stats (prefer Understat for depth)
                COALESCE(u.xg, f.xg) AS xg,
                COALESCE(u.xa, f.xg_assist) AS xa,
                COALESCE(u.npxg, f.npxg) AS npxg,
                u.xg_chain,
                u.xg_buildup,
                u.xg_overperformance,
                u.xa_overperformance,

                -- FBref detailed stats
                f.shots,
                f.shots_on_target,
                f.passes_completed,
                f.passes,
                f.passes_pct,
                f.progressive_passes,
                f.progressive_carries,
                f.tackles,
                f.interceptions,
                f.blocks,
                f.touches,
                f.take_ons_won,

                -- Source IDs for tracing
                f.player_id AS fbref_id,
                t.player_id AS transfermarkt_id,
                u.player_id AS understat_id

            FROM fbref_players f
            FULL OUTER JOIN transfermarkt_players t
                ON LOWER(f.name) = LOWER(t.name)
                AND LOWER(f.team) = LOWER(t.team)
                AND f.league = t.league
                AND f.season = t.season
            FULL OUTER JOIN understat_players u
                ON LOWER(COALESCE(f.name, t.name)) = LOWER(u.name)
                AND LOWER(COALESCE(f.team, t.team)) = LOWER(u.team)
                AND COALESCE(f.league, t.league) = u.league
                AND COALESCE(f.season, t.season) = u.season
        """)

    def insert_fbref_players(self, players: list[dict[str, Any]]) -> int:
        """Insert FBref player records.

        Uses INSERT OR REPLACE for upsert behavior.

        Args:
            players: List of player dictionaries from FBrefScraper

        Returns:
            Number of records inserted
        """
        if not players:
            return 0

        df = pd.DataFrame(players)
        df["scraped_at"] = datetime.now()

        # Get existing columns in table
        table_cols = self._get_table_columns("fbref_players")

        # Only keep columns that exist in the table
        cols_to_insert = [c for c in df.columns if c in table_cols]
        df = df[cols_to_insert]

        # Build column list for INSERT
        col_names = ", ".join(cols_to_insert)

        self.conn.execute(f"""
            INSERT OR REPLACE INTO fbref_players ({col_names})
            SELECT {col_names} FROM df
        """)

        logger.info(f"Inserted {len(players)} FBref players")
        return len(players)

    def insert_transfermarkt_players(self, players: list[dict[str, Any]]) -> int:
        """Insert Transfermarkt player records.

        Args:
            players: List of player dictionaries from TransfermarktScraper

        Returns:
            Number of records inserted
        """
        if not players:
            return 0

        df = pd.DataFrame(players)
        df["scraped_at"] = datetime.now()

        table_cols = self._get_table_columns("transfermarkt_players")
        cols_to_insert = [c for c in df.columns if c in table_cols]
        df = df[cols_to_insert]

        col_names = ", ".join(cols_to_insert)

        self.conn.execute(f"""
            INSERT OR REPLACE INTO transfermarkt_players ({col_names})
            SELECT {col_names} FROM df
        """)

        logger.info(f"Inserted {len(players)} Transfermarkt players")
        return len(players)

    def insert_understat_players(self, players: list[dict[str, Any]]) -> int:
        """Insert Understat player records.

        Args:
            players: List of player dictionaries from UnderstatScraper

        Returns:
            Number of records inserted
        """
        if not players:
            return 0

        df = pd.DataFrame(players)
        df["scraped_at"] = datetime.now()

        table_cols = self._get_table_columns("understat_players")
        cols_to_insert = [c for c in df.columns if c in table_cols]
        df = df[cols_to_insert]

        col_names = ", ".join(cols_to_insert)

        self.conn.execute(f"""
            INSERT OR REPLACE INTO understat_players ({col_names})
            SELECT {col_names} FROM df
        """)

        logger.info(f"Inserted {len(players)} Understat players")
        return len(players)

    def _get_table_columns(self, table_name: str) -> set[str]:
        """Get column names for a table."""
        result = self.conn.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """).fetchall()
        return {row[0] for row in result}

    def get_fbref_players(
        self, league: str | None = None, season: str | None = None
    ) -> pd.DataFrame:
        """Get FBref players with optional filtering.

        Args:
            league: Filter by league (optional)
            season: Filter by season (optional)

        Returns:
            DataFrame of FBref player records
        """
        query = "SELECT * FROM fbref_players WHERE 1=1"
        params = []

        if league:
            query += " AND league = ?"
            params.append(league)
        if season:
            query += " AND season = ?"
            params.append(season)

        return self.conn.execute(query, params).df()

    def get_transfermarkt_players(
        self, league: str | None = None, season: str | None = None
    ) -> pd.DataFrame:
        """Get Transfermarkt players with optional filtering."""
        query = "SELECT * FROM transfermarkt_players WHERE 1=1"
        params = []

        if league:
            query += " AND league = ?"
            params.append(league)
        if season:
            query += " AND season = ?"
            params.append(season)

        return self.conn.execute(query, params).df()

    def get_understat_players(
        self, league: str | None = None, season: str | None = None
    ) -> pd.DataFrame:
        """Get Understat players with optional filtering."""
        query = "SELECT * FROM understat_players WHERE 1=1"
        params = []

        if league:
            query += " AND league = ?"
            params.append(league)
        if season:
            query += " AND season = ?"
            params.append(season)

        return self.conn.execute(query, params).df()

    def get_unified_players(
        self, league: str | None = None, season: str | None = None
    ) -> pd.DataFrame:
        """Get unified player view with data from all sources merged.

        Args:
            league: Filter by league (optional)
            season: Filter by season (optional)

        Returns:
            DataFrame with merged player data
        """
        query = "SELECT * FROM players_unified WHERE 1=1"
        params = []

        if league:
            query += " AND league = ?"
            params.append(league)
        if season:
            query += " AND season = ?"
            params.append(season)

        return self.conn.execute(query, params).df()

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with row counts and metadata
        """
        fbref_count = self.conn.execute(
            "SELECT COUNT(*) FROM fbref_players"
        ).fetchone()[0]
        tm_count = self.conn.execute(
            "SELECT COUNT(*) FROM transfermarkt_players"
        ).fetchone()[0]
        us_count = self.conn.execute(
            "SELECT COUNT(*) FROM understat_players"
        ).fetchone()[0]

        leagues = self.conn.execute("""
            SELECT DISTINCT league FROM (
                SELECT league FROM fbref_players
                UNION
                SELECT league FROM transfermarkt_players
                UNION
                SELECT league FROM understat_players
            )
        """).fetchall()

        seasons = self.conn.execute("""
            SELECT DISTINCT season FROM (
                SELECT season FROM fbref_players
                UNION
                SELECT season FROM transfermarkt_players
                UNION
                SELECT season FROM understat_players
            )
        """).fetchall()

        return {
            "fbref_players": fbref_count,
            "transfermarkt_players": tm_count,
            "understat_players": us_count,
            "total_records": fbref_count + tm_count + us_count,
            "leagues": [r[0] for r in leagues],
            "seasons": [r[0] for r in seasons],
            "db_path": str(self.db_path),
        }

    def execute(self, query: str, params: list | None = None):
        """Execute raw SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            DuckDB query result
        """
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
