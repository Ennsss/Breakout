"""CLI for data pipeline operations.

Usage:
    python -m src.data.cli scrape --league eredivisie --season 2023-2024
    python -m src.data.cli scrape-all --season 2023-2024 --priority 1
    python -m src.data.cli validate --league eredivisie --season 2023-2024
    python -m src.data.cli stats
"""

import argparse
import logging
import sys

from .pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def cmd_scrape(args):
    """Handle scrape command."""
    with DataPipeline(
        db_path=args.db,
        cache_dir=args.cache,
        rate_limit=args.rate_limit,
    ) as pipeline:
        result = pipeline.scrape_league_season(
            league=args.league,
            season=args.season,
            sources=args.sources,
        )
        print(f"\n{result}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
            return 1
        return 0


def cmd_scrape_all(args):
    """Handle scrape-all command."""
    with DataPipeline(
        db_path=args.db,
        cache_dir=args.cache,
        rate_limit=args.rate_limit,
    ) as pipeline:
        results = pipeline.scrape_feeder_leagues(
            season=args.season,
            priority=args.priority,
            sources=args.sources,
        )

        print("\n=== Scrape Summary ===")
        total_records = 0
        total_errors = 0
        for result in results:
            print(result)
            total_records += result.total_records
            total_errors += len(result.errors)

        print(f"\nTotal: {total_records} records, {total_errors} errors")
        return 1 if total_errors > 0 else 0


def cmd_validate(args):
    """Handle validate command."""
    with DataPipeline(db_path=args.db) as pipeline:
        if args.league and args.season:
            report = pipeline.validate_data(args.league, args.season)
            print(report)
            if report.warnings:
                print("\nWarnings:")
                for warning in report.warnings:
                    print(f"  - {warning}")
        else:
            # Validate all data in database
            stats = pipeline.get_scrape_summary()
            for league in stats.get("leagues", []):
                for season in stats.get("seasons", []):
                    report = pipeline.validate_data(league, season)
                    print(report)
                    print()
    return 0


def cmd_stats(args):
    """Handle stats command."""
    with DataPipeline(db_path=args.db) as pipeline:
        stats = pipeline.get_scrape_summary()

        print("=== Database Statistics ===")
        print(f"Database: {stats['db_path']}")
        print(f"\nRecords by source:")
        print(f"  FBref:        {stats['fbref_players']:,}")
        print(f"  Transfermarkt: {stats['transfermarkt_players']:,}")
        print(f"  Understat:    {stats['understat_players']:,}")
        print(f"  Total:        {stats['total_records']:,}")

        if stats["leagues"]:
            print(f"\nLeagues: {', '.join(sorted(stats['leagues']))}")
        if stats["seasons"]:
            print(f"Seasons: {', '.join(sorted(stats['seasons']))}")
    return 0


def cmd_leagues(args):
    """Handle leagues command."""
    with DataPipeline() as pipeline:
        if args.source:
            leagues = pipeline.get_supported_leagues(args.source)
            print(f"Leagues supported by {args.source}:")
        else:
            leagues = pipeline.get_supported_leagues()
            print("All supported leagues:")

        for league in leagues:
            print(f"  - {league}")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hidden Gem Finder - Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--db",
        default="data/players.duckdb",
        help="Path to DuckDB database (default: data/players.duckdb)",
    )
    parser.add_argument(
        "--cache",
        default="data/raw",
        help="Cache directory for scraped HTML (default: data/raw)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=3.0,
        help="Seconds between requests (default: 3.0)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scrape command
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape a single league-season",
    )
    scrape_parser.add_argument(
        "--league", "-l",
        required=True,
        help="League name (e.g., eredivisie, premier-league)",
    )
    scrape_parser.add_argument(
        "--season", "-s",
        required=True,
        help="Season (e.g., 2023-2024)",
    )
    scrape_parser.add_argument(
        "--sources",
        nargs="+",
        default=["fbref", "transfermarkt", "understat"],
        help="Data sources to scrape (default: all)",
    )

    # scrape-all command
    scrape_all_parser = subparsers.add_parser(
        "scrape-all",
        help="Scrape all feeder leagues for a season",
    )
    scrape_all_parser.add_argument(
        "--season", "-s",
        required=True,
        help="Season (e.g., 2023-2024)",
    )
    scrape_all_parser.add_argument(
        "--priority",
        type=int,
        choices=[1, 2],
        default=1,
        help="League priority (1=top feeders, 2=all feeders)",
    )
    scrape_all_parser.add_argument(
        "--sources",
        nargs="+",
        default=["fbref", "transfermarkt", "understat"],
        help="Data sources to scrape (default: all)",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate data quality",
    )
    validate_parser.add_argument(
        "--league", "-l",
        help="League name (optional, validates all if omitted)",
    )
    validate_parser.add_argument(
        "--season", "-s",
        help="Season (optional, validates all if omitted)",
    )

    # stats command
    subparsers.add_parser(
        "stats",
        help="Show database statistics",
    )

    # leagues command
    leagues_parser = subparsers.add_parser(
        "leagues",
        help="List supported leagues",
    )
    leagues_parser.add_argument(
        "--source",
        choices=["fbref", "transfermarkt", "understat"],
        help="Filter by data source",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "scrape": cmd_scrape,
        "scrape-all": cmd_scrape_all,
        "validate": cmd_validate,
        "stats": cmd_stats,
        "leagues": cmd_leagues,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
