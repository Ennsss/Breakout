"""Understat scraper for xG/xA statistics."""

import json
import logging
import re
from typing import Any

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class UnderstatScraper(BaseScraper):
    """Scraper for Understat.com xG/xA statistics.

    Understat provides detailed expected goals data for 6 European leagues:
    - Premier League (EPL)
    - La Liga
    - Bundesliga
    - Serie A
    - Ligue 1
    - Eredivisie

    Data is embedded as JSON in JavaScript within HTML pages.
    """

    BASE_URL = "https://understat.com"

    # Map our kebab-case league keys to Understat league names
    # None means the league is not available on Understat
    LEAGUE_IDS = {
        # Available on Understat
        "eredivisie": "Eredivisie",
        "premier-league": "EPL",
        "la-liga": "La_liga",
        "bundesliga": "Bundesliga",
        "serie-a": "Serie_A",
        "ligue-1": "Ligue_1",
        # Not available on Understat
        "primeira-liga": None,
        "belgian-pro-league": None,
        "championship": None,
        "serie-b": None,
        "ligue-2": None,
        "austrian-bundesliga": None,
        "scottish-premiership": None,
    }

    # Position code normalization
    POSITION_MAP = {
        "F": "FW",
        "M": "MF",
        "D": "DF",
        "GK": "GK",
        "S": "FW",  # Sub/Striker -> Forward
    }

    @property
    def source_name(self) -> str:
        return "understat"

    def _convert_season_format(self, season: str) -> str:
        """Convert '2023-2024' or '2023-24' to '2023' (start year)."""
        return season.split("-")[0]

    def _build_league_url(self, league: str, season: str) -> str:
        """Build URL for league season page."""
        league_key = league.lower().replace(" ", "-")
        league_name = self.LEAGUE_IDS.get(league_key)

        if league_name is None:
            available = [k for k, v in self.LEAGUE_IDS.items() if v is not None]
            raise ValueError(
                f"League '{league}' not available on Understat. "
                f"Available leagues: {available}"
            )

        season_year = self._convert_season_format(season)
        return f"{self.BASE_URL}/league/{league_name}/{season_year}"

    def _decode_json_string(self, encoded: str) -> str:
        """Decode unicode/hex escaped string from Understat."""
        try:
            return encoded.encode("utf-8").decode("unicode_escape")
        except Exception:
            return encoded

    def _extract_players_data(self, html: str) -> list[dict]:
        """Extract playersData JSON from page HTML."""
        pattern = r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)

        if match is None:
            logger.warning("Could not find playersData in page")
            return []

        encoded_string = match.group(1)
        decoded_string = self._decode_json_string(encoded_string)

        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse playersData JSON: {e}")
            return []

    def _safe_float(self, val: Any, default: float | None = None) -> float | None:
        """Safely convert value to float."""
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _safe_int(self, val: Any, default: int | None = None) -> int | None:
        """Safely convert value to int."""
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _parse_player_record(
        self, raw: dict, league: str, season: str
    ) -> dict[str, Any]:
        """Parse raw Understat player record to normalized format."""
        minutes = self._safe_int(raw.get("time"), 0)
        minutes_90s = minutes / 90.0 if minutes > 0 else 0.0

        goals = self._safe_int(raw.get("goals"), 0)
        assists = self._safe_int(raw.get("assists"), 0)
        xg = self._safe_float(raw.get("xG"), 0.0)
        xa = self._safe_float(raw.get("xA"), 0.0)
        npxg = self._safe_float(raw.get("npxG"), 0.0)

        raw_position = raw.get("position", "")
        position = self.POSITION_MAP.get(raw_position, raw_position)

        return {
            "player_id": str(raw.get("id", "")),
            "name": raw.get("player_name", ""),
            "position": position,
            "team": raw.get("team_title"),
            "games": self._safe_int(raw.get("games")),
            "minutes": minutes,
            "minutes_90s": round(minutes_90s, 2),
            "goals": goals,
            "assists": assists,
            "npg": self._safe_int(raw.get("npg")),
            "xg": xg,
            "xa": xa,
            "npxg": npxg,
            "xg_chain": self._safe_float(raw.get("xGChain")),
            "xg_buildup": self._safe_float(raw.get("xGBuildup")),
            "xg_per90": round(xg / minutes_90s, 2) if minutes_90s > 0 else None,
            "xa_per90": round(xa / minutes_90s, 2) if minutes_90s > 0 else None,
            "npxg_per90": round(npxg / minutes_90s, 2) if minutes_90s > 0 else None,
            "goals_per90": round(goals / minutes_90s, 2) if minutes_90s > 0 else None,
            "assists_per90": round(assists / minutes_90s, 2) if minutes_90s > 0 else None,
            "shots": self._safe_int(raw.get("shots")),
            "key_passes": self._safe_int(raw.get("key_passes")),
            "yellow_cards": self._safe_int(raw.get("yellow_cards")),
            "red_cards": self._safe_int(raw.get("red_cards")),
            "xg_overperformance": round(goals - xg, 2) if xg is not None else None,
            "xa_overperformance": round(assists - xa, 2) if xa is not None else None,
            "league": league,
            "season": season,
            "source": "understat",
        }

    def scrape_league_season(
        self,
        league: str,
        season: str,
    ) -> list[dict[str, Any]]:
        """Scrape all player xG/xA stats for a league season."""
        league_key = league.lower().replace(" ", "-")

        if league_key not in self.LEAGUE_IDS:
            available = [k for k, v in self.LEAGUE_IDS.items() if v is not None]
            raise ValueError(
                f"Unknown league: {league}. "
                f"Available on Understat: {available}"
            )

        if self.LEAGUE_IDS[league_key] is None:
            available = [k for k, v in self.LEAGUE_IDS.items() if v is not None]
            raise ValueError(
                f"League '{league}' not available on Understat. "
                f"Available leagues: {available}"
            )

        logger.info(f"Scraping Understat: {league} {season}")

        url = self._build_league_url(league_key, season)
        html = self.fetch(url)

        raw_players = self._extract_players_data(html)
        logger.info(f"Found {len(raw_players)} players in raw data")

        players = []
        for raw_player in raw_players:
            try:
                player_data = self._parse_player_record(raw_player, league, season)
                players.append(player_data)
            except Exception as e:
                logger.warning(
                    f"Error parsing player {raw_player.get('player_name', 'unknown')}: {e}"
                )
                continue

        logger.info(f"Scraped {len(players)} players for {league} {season}")
        return players
