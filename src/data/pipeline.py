"""Data pipeline for orchestrating scraping from all sources."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.scrapers import FBrefScraper, TransfermarktScraper, UnderstatScraper
from src.storage import PlayerDatabase

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result of a scraping operation for a single league-season."""

    league: str
    season: str
    fbref_count: int = 0
    transfermarkt_count: int = 0
    understat_count: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """True if no errors occurred."""
        return len(self.errors) == 0

    @property
    def total_records(self) -> int:
        """Total records inserted across all sources."""
        return self.fbref_count + self.transfermarkt_count + self.understat_count

    def __str__(self) -> str:
        status = "OK" if self.success else f"ERRORS: {len(self.errors)}"
        return (
            f"{self.league} {self.season}: "
            f"FBref={self.fbref_count}, TM={self.transfermarkt_count}, "
            f"US={self.understat_count} [{status}]"
        )


@dataclass
class ValidationReport:
    """Report on data quality for a league-season."""

    league: str
    season: str
    fbref_players: int = 0
    transfermarkt_players: int = 0
    understat_players: int = 0
    unified_players: int = 0
    match_rate: float = 0.0
    missing_market_values: int = 0
    missing_xg_data: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Validation: {self.league} {self.season}\n"
            f"  FBref: {self.fbref_players} | TM: {self.transfermarkt_players} | "
            f"US: {self.understat_players}\n"
            f"  Unified: {self.unified_players} | Match rate: {self.match_rate:.1%}\n"
            f"  Missing market values: {self.missing_market_values}\n"
            f"  Missing xG data: {self.missing_xg_data}\n"
            f"  Warnings: {len(self.warnings)}"
        )


class DataPipeline:
    """Orchestrates data collection from all sources into DuckDB.

    Handles:
    - Scraping from FBref, Transfermarkt, and Understat
    - Storing data in DuckDB
    - Validating data quality
    - Graceful handling of source limitations (e.g., Understat only has 6 leagues)
    """

    # Leagues supported by FBref and Transfermarkt
    ALL_LEAGUES = [
        # Feeder leagues (priority 1)
        "eredivisie",
        "primeira-liga",
        "belgian-pro-league",
        "championship",
        # Feeder leagues (priority 2)
        "serie-b",
        "ligue-2",
        "austrian-bundesliga",
        "scottish-premiership",
        # Target leagues (top 5)
        "premier-league",
        "la-liga",
        "bundesliga",
        "serie-a",
        "ligue-1",
    ]

    # Priority groupings for feeder leagues
    PRIORITY_1_LEAGUES = [
        "eredivisie",
        "primeira-liga",
        "belgian-pro-league",
        "championship",
    ]

    PRIORITY_2_LEAGUES = [
        "serie-b",
        "ligue-2",
        "austrian-bundesliga",
        "scottish-premiership",
    ]

    TARGET_LEAGUES = [
        "premier-league",
        "la-liga",
        "bundesliga",
        "serie-a",
        "ligue-1",
    ]

    # Understat only supports 6 leagues
    UNDERSTAT_LEAGUES = [
        "eredivisie",
        "premier-league",
        "la-liga",
        "bundesliga",
        "serie-a",
        "ligue-1",
    ]

    def __init__(
        self,
        db_path: str | Path = "data/players.duckdb",
        cache_dir: str | Path = "data/raw",
        rate_limit: float = 3.0,
    ):
        """Initialize pipeline with database and scrapers.

        Args:
            db_path: Path to DuckDB database file
            cache_dir: Directory for caching scraped HTML
            rate_limit: Seconds between requests to each source
        """
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)
        self.rate_limit = rate_limit

        # Initialize database
        self.db = PlayerDatabase(db_path)

        # Initialize scrapers
        self.fbref = FBrefScraper(cache_dir=cache_dir, rate_limit=rate_limit)
        self.transfermarkt = TransfermarktScraper(
            cache_dir=cache_dir, rate_limit=rate_limit
        )
        self.understat = UnderstatScraper(cache_dir=cache_dir, rate_limit=rate_limit)

        logger.info(f"Pipeline initialized: db={db_path}, cache={cache_dir}")

    def scrape_league_season(
        self,
        league: str,
        season: str,
        sources: list[str] | None = None,
    ) -> ScrapeResult:
        """Scrape data for a single league-season from specified sources.

        Args:
            league: League name (e.g., "eredivisie")
            season: Season string (e.g., "2023-2024")
            sources: List of sources to scrape. Default: all sources.
                    Options: "fbref", "transfermarkt", "understat"

        Returns:
            ScrapeResult with counts and any errors
        """
        if sources is None:
            sources = ["fbref", "transfermarkt", "understat"]

        start_time = time.time()
        result = ScrapeResult(league=league, season=season)

        # Validate league
        if league not in self.ALL_LEAGUES:
            result.errors.append(f"Unknown league: {league}")
            return result

        # Scrape FBref
        if "fbref" in sources:
            try:
                logger.info(f"Scraping FBref: {league} {season}")
                players = self.fbref.scrape_league_season(league, season)
                result.fbref_count = self.db.insert_fbref_players(players)
                logger.info(f"FBref: inserted {result.fbref_count} players")
            except Exception as e:
                error_msg = f"FBref error: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Scrape Transfermarkt
        if "transfermarkt" in sources:
            try:
                logger.info(f"Scraping Transfermarkt: {league} {season}")
                players = self.transfermarkt.scrape_league_season(league, season)
                result.transfermarkt_count = self.db.insert_transfermarkt_players(
                    players
                )
                logger.info(f"Transfermarkt: inserted {result.transfermarkt_count} players")
            except Exception as e:
                error_msg = f"Transfermarkt error: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Scrape Understat (if available for this league)
        if "understat" in sources:
            if league in self.UNDERSTAT_LEAGUES:
                try:
                    logger.info(f"Scraping Understat: {league} {season}")
                    players = self.understat.scrape_league_season(league, season)
                    result.understat_count = self.db.insert_understat_players(players)
                    logger.info(f"Understat: inserted {result.understat_count} players")
                except Exception as e:
                    error_msg = f"Understat error: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            else:
                logger.info(f"Understat not available for {league}, skipping")

        result.duration_seconds = time.time() - start_time
        logger.info(f"Completed {league} {season} in {result.duration_seconds:.1f}s")

        return result

    def scrape_multiple_leagues(
        self,
        leagues: list[str],
        season: str,
        sources: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[ScrapeResult]:
        """Scrape multiple leagues for a single season.

        Args:
            leagues: List of league names
            season: Season string
            sources: Sources to scrape (default: all)
            show_progress: Show tqdm progress bar

        Returns:
            List of ScrapeResult objects
        """
        results = []
        iterator = tqdm(leagues, desc=f"Scraping {season}") if show_progress else leagues

        for league in iterator:
            result = self.scrape_league_season(league, season, sources)
            results.append(result)

            if show_progress:
                iterator.set_postfix(
                    {"total": result.total_records, "errors": len(result.errors)}
                )

        return results

    def scrape_multiple_seasons(
        self,
        league: str,
        seasons: list[str],
        sources: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[ScrapeResult]:
        """Scrape a single league across multiple seasons.

        Args:
            league: League name
            seasons: List of season strings
            sources: Sources to scrape (default: all)
            show_progress: Show tqdm progress bar

        Returns:
            List of ScrapeResult objects
        """
        results = []
        iterator = tqdm(seasons, desc=f"Scraping {league}") if show_progress else seasons

        for season in iterator:
            result = self.scrape_league_season(league, season, sources)
            results.append(result)

            if show_progress:
                iterator.set_postfix(
                    {"total": result.total_records, "errors": len(result.errors)}
                )

        return results

    def scrape_feeder_leagues(
        self,
        season: str,
        priority: int = 1,
        sources: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[ScrapeResult]:
        """Scrape all feeder leagues (source leagues for talent scouting).

        Args:
            season: Season string
            priority: 1 for top feeder leagues, 2 for all feeder leagues
            sources: Sources to scrape (default: all)
            show_progress: Show progress bar

        Returns:
            List of ScrapeResult objects
        """
        if priority == 1:
            leagues = self.PRIORITY_1_LEAGUES
        else:
            leagues = self.PRIORITY_1_LEAGUES + self.PRIORITY_2_LEAGUES

        return self.scrape_multiple_leagues(leagues, season, sources, show_progress)

    def validate_data(self, league: str, season: str) -> ValidationReport:
        """Validate data quality for a league-season.

        Checks:
        - Record counts per source
        - Match rate in unified view
        - Missing market values
        - Missing xG data

        Args:
            league: League name
            season: Season string

        Returns:
            ValidationReport with quality metrics
        """
        report = ValidationReport(league=league, season=season)

        # Get counts per source
        fbref_df = self.db.get_fbref_players(league=league, season=season)
        tm_df = self.db.get_transfermarkt_players(league=league, season=season)
        us_df = self.db.get_understat_players(league=league, season=season)
        unified_df = self.db.get_unified_players(league=league, season=season)

        report.fbref_players = len(fbref_df)
        report.transfermarkt_players = len(tm_df)
        report.understat_players = len(us_df)
        report.unified_players = len(unified_df)

        # Calculate match rate (players found in multiple sources)
        if report.unified_players > 0:
            # Count players with IDs from multiple sources
            multi_source = unified_df[
                (unified_df["fbref_id"].notna().astype(int) +
                 unified_df["transfermarkt_id"].notna().astype(int) +
                 unified_df["understat_id"].notna().astype(int)) >= 2
            ]
            report.match_rate = len(multi_source) / report.unified_players

        # Check for missing data
        if report.unified_players > 0:
            report.missing_market_values = unified_df["market_value_eur"].isna().sum()
            report.missing_xg_data = unified_df["xg"].isna().sum()

        # Generate warnings
        if report.fbref_players == 0:
            report.warnings.append("No FBref data found")
        if report.transfermarkt_players == 0:
            report.warnings.append("No Transfermarkt data found")
        if league in self.UNDERSTAT_LEAGUES and report.understat_players == 0:
            report.warnings.append("No Understat data found (expected for this league)")
        if report.match_rate < 0.5:
            report.warnings.append(f"Low match rate: {report.match_rate:.1%}")
        if report.missing_market_values > report.unified_players * 0.5:
            report.warnings.append("More than 50% missing market values")

        return report

    def get_supported_leagues(self, source: str | None = None) -> list[str]:
        """Get list of supported leagues, optionally filtered by source.

        Args:
            source: Filter by source ("fbref", "transfermarkt", "understat")

        Returns:
            List of league names
        """
        if source is None:
            return self.ALL_LEAGUES
        elif source == "understat":
            return self.UNDERSTAT_LEAGUES
        else:
            return self.ALL_LEAGUES

    def get_scrape_summary(self) -> dict[str, Any]:
        """Get summary of all scraped data.

        Returns:
            Dictionary with database statistics
        """
        return self.db.get_stats()

    def close(self) -> None:
        """Close database connection."""
        self.db.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
