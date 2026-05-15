#!/usr/bin/env python3
"""Setup script for pending tasks from RECIPES.md

Tasks:
1. Authenticate with learningsystems Slack workspace
2. Authenticate with iMessages agent
3. Create cron job to poll kisssorcar channel
4. Send test message to 5102893391
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kiss.agents.third_party_agents.cron_manager_daemon import CronClient, start_daemon, daemon_status


def setup_cron_job() -> None:
    """Create a cron job for the kisssorcar channel poller."""
    print("\n" + "=" * 70)
    print("Setting up cron job for kisssorcar channel poller...")
    print("=" * 70)

    # Check daemon status
    print(f"Daemon status: {daemon_status()}")

    # Try to start the daemon if not running
    try:
        print("Ensuring cron manager daemon is running...")
        start_daemon(foreground=False)
        print(f"✓ {daemon_status()}")
    except Exception as e:
        print(f"  Note: {e}")

    # Create the cron job
    # Runs every minute, polls for 57s at 3s intervals
    cron_entry = (
        "* * * * * "
        "KISS_SLACK_WORKSPACE=learningsystems "
        "KISS_SLACK_USER=ksen "
        "KISS_SLACK_CHANNEL=kisssorcar "
        "/usr/bin/python3 -m kiss.agents.third_party_agents.slack_channel_sorcar_poller"
    )

    try:
        client = CronClient()
        job_id = client.add_job(cron_entry)
        print(f"\n✓ Cron job created: {job_id}")
        print(f"\nCron entry:")
        print(f"  {cron_entry}")

        # List all jobs
        jobs = client.list_jobs()
        print(f"\nTotal KISS jobs installed: {len(jobs)}")
        for job in jobs:
            print(f"  - {job['id']}: {job['schedule']} {job['command'][:50]}...")
    except Exception as e:
        print(f"✗ Failed to create cron job: {e}")
        raise


def show_next_steps() -> None:
    """Show next steps for manual authentication."""
    print("\n" + "=" * 70)
    print("NEXT STEPS - Manual Authentication Required")
    print("=" * 70)

    print("\n1. SLACK AUTHENTICATION (learningsystems workspace):")
    print("   Run this command in a terminal:")
    print("   $ python -m kiss.agents.third_party_agents.slack_agent --workspace learningsystems")
    print("   Follow the prompts to authenticate with Slack.")

    print("\n2. IMESSAGE AUTHENTICATION:")
    print("   On macOS, run this command:")
    print("   $ python -m kiss.agents.third_party_agents.imessage_agent")
    print("   OR use BlueBubbles:")
    print("   $ python -m kiss.agents.third_party_agents.bluebubbles_agent")

    print("\n3. SEND TEST MESSAGE:")
    print("   After authentication, run:")
    print("   $ python -m kiss.agents.third_party_agents.imessage_agent")
    print("   Then use prompt: 'send \"Hello\" to 5102893391'")

    print("\n4. MONITOR THE POLLER:")
    print("   View logs at: ~/.kiss/slack_channel_sorcar_poller/poller.log")
    print("   Check cron logs: log stream --predicate 'process == \"cron\"'")


def main() -> None:
    """Run setup tasks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " KISS Sorcar Recipe Setup ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    # Task 1: Setup cron job (automated)
    try:
        setup_cron_job()
    except Exception as e:
        print(f"\n✗ Cron job setup failed: {e}")
        sys.exit(1)

    # Tasks 2-4: Show manual steps
    show_next_steps()

    print("\n" + "=" * 70)
    print("Setup initiated successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
