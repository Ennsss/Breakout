"""Tests for DuckDB storage layer."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.storage import PlayerDatabase


class TestPlayerDatabase:
    """Tests for PlayerDatabase class."""

    @pytest.fixture
    def db(self):
        """Create an in-memory database for testing."""
        database = PlayerDatabase(":memory:")
        yield database
        database.close()

    @pytest.fixture
    def sample_fbref_players(self):
        """Sample FBref player data."""
        return [
            {
                "player_id": "abc12345",
                "name": "Cody Gakpo",
                "position": "FW,MF",
                "team": "PSV Eindhoven",
                "nationality": "Netherlands",
                "age": 23,
                "games": 34,
                "minutes": 2856,
                "goals": 21,
                "assists": 15,
                "xg": 15.8,
                "npxg": 12.4,
                "xg_assist": 10.2,
                "shots": 98,
                "passes_completed": 1200,
                "passes_pct": 85.5,
                "progressive_passes": 145,
                "tackles": 32,
                "interceptions": 18,
                "league": "eredivisie",
                "season": "2023-2024",
                "source": "fbref",
            },
            {
                "player_id": "def67890",
                "name": "Test Midfielder",
                "position": "MF",
                "team": "Ajax",
                "nationality": "Netherlands",
                "age": 25,
                "games": 28,
                "minutes": 2145,
                "goals": 5,
                "assists": 8,
                "xg": 4.2,
                "league": "eredivisie",
                "season": "2023-2024",
                "source": "fbref",
            },
        ]

    @pytest.fixture
    def sample_transfermarkt_players(self):
        """Sample Transfermarkt player data."""
        return [
            {
                "player_id": "363205",
                "player_slug": "cody-gakpo",
                "name": "Cody Gakpo",
                "position": "Left Winger",
                "team": "PSV Eindhoven",
                "age": 23,
                "nationality": "Netherlands",
                "market_value_eur": 45_000_000,
                "league": "eredivisie",
                "season": "2023-2024",
                "source": "transfermarkt",
            },
            {
                "player_id": "123456",
                "player_slug": "test-midfielder",
                "name": "Test Midfielder",
                "position": "Central Midfield",
                "team": "Ajax",
                "age": 25,
                "market_value_eur": 15_000_000,
                "league": "eredivisie",
                "season": "2023-2024",
                "source": "transfermarkt",
            },
        ]

    @pytest.fixture
    def sample_understat_players(self):
        """Sample Understat player data."""
        return [
            {
                "player_id": "1234",
                "name": "Cody Gakpo",
                "position": "FW",
                "team": "PSV Eindhoven",
                "games": 34,
                "minutes": 2856,
                "goals": 21,
                "assists": 15,
                "xg": 15.8,
                "xa": 10.2,
                "npxg": 12.4,
                "xg_chain": 22.5,
                "xg_buildup": 8.3,
                "xg_per90": 0.50,
                "xa_per90": 0.32,
                "xg_overperformance": 5.2,
                "xa_overperformance": 4.8,
                "league": "eredivisie",
                "season": "2023-2024",
                "source": "understat",
            },
        ]

    def test_database_creation(self, db):
        """Test that database is created with all tables."""
        # Check tables exist
        tables = db.conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
            AND table_type = 'BASE TABLE'
        """).fetchall()
        table_names = {t[0] for t in tables}

        assert "fbref_players" in table_names
        assert "transfermarkt_players" in table_names
        assert "understat_players" in table_names

    def test_unified_view_exists(self, db):
        """Test that unified view is created."""
        views = db.conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
            AND table_type = 'VIEW'
        """).fetchall()
        view_names = {v[0] for v in views}

        assert "players_unified" in view_names

    def test_insert_fbref_players(self, db, sample_fbref_players):
        """Test inserting FBref players."""
        count = db.insert_fbref_players(sample_fbref_players)

        assert count == 2

        df = db.get_fbref_players()
        assert len(df) == 2
        assert "Cody Gakpo" in df["name"].values

    def test_insert_transfermarkt_players(self, db, sample_transfermarkt_players):
        """Test inserting Transfermarkt players."""
        count = db.insert_transfermarkt_players(sample_transfermarkt_players)

        assert count == 2

        df = db.get_transfermarkt_players()
        assert len(df) == 2
        assert df[df["name"] == "Cody Gakpo"]["market_value_eur"].iloc[0] == 45_000_000

    def test_insert_understat_players(self, db, sample_understat_players):
        """Test inserting Understat players."""
        count = db.insert_understat_players(sample_understat_players)

        assert count == 1

        df = db.get_understat_players()
        assert len(df) == 1
        assert df.iloc[0]["xg_chain"] == 22.5

    def test_insert_empty_list(self, db):
        """Test inserting empty list returns 0."""
        assert db.insert_fbref_players([]) == 0
        assert db.insert_transfermarkt_players([]) == 0
        assert db.insert_understat_players([]) == 0

    def test_upsert_behavior(self, db, sample_fbref_players):
        """Test that duplicate inserts update existing records."""
        db.insert_fbref_players(sample_fbref_players)

        # Modify and re-insert
        updated_players = sample_fbref_players.copy()
        updated_players[0]["goals"] = 25  # Changed from 21

        db.insert_fbref_players(updated_players)

        df = db.get_fbref_players()
        assert len(df) == 2  # Still only 2 records
        gakpo = df[df["name"] == "Cody Gakpo"].iloc[0]
        assert gakpo["goals"] == 25  # Updated value

    def test_filter_by_league(self, db, sample_fbref_players):
        """Test filtering by league."""
        db.insert_fbref_players(sample_fbref_players)

        df = db.get_fbref_players(league="eredivisie")
        assert len(df) == 2

        df = db.get_fbref_players(league="premier-league")
        assert len(df) == 0

    def test_filter_by_season(self, db, sample_fbref_players):
        """Test filtering by season."""
        db.insert_fbref_players(sample_fbref_players)

        df = db.get_fbref_players(season="2023-2024")
        assert len(df) == 2

        df = db.get_fbref_players(season="2022-2023")
        assert len(df) == 0

    def test_filter_by_league_and_season(self, db, sample_fbref_players):
        """Test filtering by both league and season."""
        db.insert_fbref_players(sample_fbref_players)

        df = db.get_fbref_players(league="eredivisie", season="2023-2024")
        assert len(df) == 2

    def test_unified_view_merges_data(
        self,
        db,
        sample_fbref_players,
        sample_transfermarkt_players,
        sample_understat_players,
    ):
        """Test that unified view merges data from all sources."""
        db.insert_fbref_players(sample_fbref_players)
        db.insert_transfermarkt_players(sample_transfermarkt_players)
        db.insert_understat_players(sample_understat_players)

        df = db.get_unified_players(league="eredivisie", season="2023-2024")

        # Should have merged records
        assert len(df) > 0

        # Find Gakpo's merged record
        gakpo = df[df["name"] == "Cody Gakpo"]
        assert len(gakpo) == 1

        gakpo = gakpo.iloc[0]
        # Should have market value from Transfermarkt
        assert gakpo["market_value_eur"] == 45_000_000
        # Should have xG chain from Understat
        assert gakpo["xg_chain"] == 22.5
        # Should have passes from FBref
        assert gakpo["passes_completed"] == 1200

    def test_get_stats(
        self,
        db,
        sample_fbref_players,
        sample_transfermarkt_players,
        sample_understat_players,
    ):
        """Test database statistics."""
        db.insert_fbref_players(sample_fbref_players)
        db.insert_transfermarkt_players(sample_transfermarkt_players)
        db.insert_understat_players(sample_understat_players)

        stats = db.get_stats()

        assert stats["fbref_players"] == 2
        assert stats["transfermarkt_players"] == 2
        assert stats["understat_players"] == 1
        assert stats["total_records"] == 5
        assert "eredivisie" in stats["leagues"]
        assert "2023-2024" in stats["seasons"]

    def test_context_manager(self):
        """Test database as context manager."""
        with PlayerDatabase(":memory:") as db:
            db.insert_fbref_players([{
                "player_id": "test",
                "name": "Test Player",
                "league": "eredivisie",
                "season": "2023-2024",
            }])
            df = db.get_fbref_players()
            assert len(df) == 1

    def test_file_based_database(self):
        """Test creating file-based database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"

            with PlayerDatabase(db_path) as db:
                db.insert_fbref_players([{
                    "player_id": "test",
                    "name": "Test Player",
                    "league": "eredivisie",
                    "season": "2023-2024",
                }])

            # File should exist
            assert db_path.exists()

            # Re-open and verify data persists
            with PlayerDatabase(db_path) as db:
                df = db.get_fbref_players()
                assert len(df) == 1

    def test_execute_raw_query(self, db, sample_fbref_players):
        """Test executing raw SQL queries."""
        db.insert_fbref_players(sample_fbref_players)

        result = db.execute(
            "SELECT name, goals FROM fbref_players WHERE goals > ?", [10]
        ).fetchall()

        assert len(result) == 1
        assert result[0][0] == "Cody Gakpo"
        assert result[0][1] == 21

    def test_empty_database_queries(self, db):
        """Test querying empty database returns empty DataFrame."""
        df = db.get_fbref_players()
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

        df = db.get_unified_players()
        assert len(df) == 0

    def test_get_stats_empty_database(self, db):
        """Test stats on empty database."""
        stats = db.get_stats()

        assert stats["fbref_players"] == 0
        assert stats["transfermarkt_players"] == 0
        assert stats["understat_players"] == 0
        assert stats["total_records"] == 0
        assert stats["leagues"] == []
        assert stats["seasons"] == []


class TestPlayerDatabaseIntegration:
    """Integration tests for PlayerDatabase with real scrapers."""

    @pytest.mark.skip(reason="Integration test - requires scraper fixtures")
    def test_insert_real_scraper_output(self, tmp_path):
        """Test inserting data from actual scrapers."""
        from src.scrapers import FBrefScraper

        scraper = FBrefScraper(cache_dir=tmp_path, rate_limit=0)
        # Would need to mock or use fixtures
        pass
