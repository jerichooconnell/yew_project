#!/usr/bin/env python3
"""
Parse yew observation reports from GitHub Issues into a training-ready CSV.

Fetches all issues with the 'observation' label from the yew_project repo,
extracts the embedded CSV data, and combines into a single file.

Usage:
    # Public repo — no token needed for reading
    python scripts/preprocessing/parse_github_observations.py

    # With token for private repos or higher rate limits
    python scripts/preprocessing/parse_github_observations.py --token ghp_xxxxx

Output:
    data/processed/github_observations.csv
    Columns: has_yew, lat, lon, issue_number, submitted_at, submitter
"""

import argparse
import csv
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


GITHUB_OWNER = 'jerichooconnell'
GITHUB_REPO  = 'yew_project'
API_BASE     = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}'


def fetch_issues(token=None, label='observation', state='all'):
    """Fetch all issues with the given label via the GitHub REST API."""
    issues = []
    page = 1
    per_page = 100

    while True:
        url = (f'{API_BASE}/issues?labels={label}&state={state}'
               f'&per_page={per_page}&page={page}')
        headers = {'Accept': 'application/vnd.github+json'}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        req = Request(url, headers=headers)
        try:
            with urlopen(req) as resp:
                data = json.loads(resp.read())
        except HTTPError as e:
            if e.code == 403:
                print(f"Rate limited. Use --token for higher limits.")
            raise

        if not data:
            break
        issues.extend(data)
        if len(data) < per_page:
            break
        page += 1

    return issues


def extract_csv_from_body(body):
    """Extract CSV data from the issue body's ``` csv code block."""
    # Match ```csv ... ``` code blocks
    pattern = r'```csv\s*\n(.*?)\n```'
    match = re.search(pattern, body, re.DOTALL)
    if not match:
        return []

    csv_text = match.group(1).strip()
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        try:
            rows.append({
                'has_yew': int(row['has_yew']),
                'lat': float(row['lat']),
                'lon': float(row['lon']),
            })
        except (ValueError, KeyError) as e:
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Parse yew observations from GitHub Issues into CSV')
    parser.add_argument('--token', type=str, default=None,
                        help='GitHub personal access token (optional, for private repos)')
    parser.add_argument('--label', type=str, default='observation',
                        help='Issue label to filter by (default: observation)')
    parser.add_argument('--output', type=str,
                        default='data/processed/github_observations.csv',
                        help='Output CSV path')
    parser.add_argument('--state', type=str, default='all',
                        choices=['open', 'closed', 'all'],
                        help='Issue state filter (default: all)')
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PARSE YEW OBSERVATIONS FROM GITHUB ISSUES")
    print("=" * 60)

    print(f"\nFetching issues from {GITHUB_OWNER}/{GITHUB_REPO}...")
    issues = fetch_issues(token=args.token, label=args.label, state=args.state)
    print(f"  Found {len(issues)} issues with label '{args.label}'")

    all_rows = []
    for issue in issues:
        number = issue['number']
        user = issue['user']['login']
        created = issue['created_at']
        body = issue.get('body', '') or ''

        rows = extract_csv_from_body(body)
        if rows:
            for r in rows:
                r['issue_number'] = number
                r['submitted_at'] = created
                r['submitter'] = user
            all_rows.extend(rows)
            print(f"  Issue #{number} ({user}, {created[:10]}): "
                  f"{len(rows)} observations")
        else:
            print(f"  Issue #{number}: no CSV data found, skipping")

    if not all_rows:
        print("\nNo observations extracted.")
        return

    # Write CSV
    fieldnames = ['has_yew', 'lat', 'lon', 'issue_number', 'submitted_at', 'submitter']
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_yew = sum(1 for r in all_rows if r['has_yew'] == 1)
    n_absent = len(all_rows) - n_yew

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total observations: {len(all_rows)}")
    print(f"  Yew present: {n_yew}")
    print(f"  Yew absent: {n_absent}")
    print(f"  Issues processed: {len(issues)}")
    print(f"  Output: {output}")


if __name__ == '__main__':
    main()
