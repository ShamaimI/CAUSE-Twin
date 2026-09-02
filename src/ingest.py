"""
CAUSE-Twin: Data Ingestion Module (Module 1)
Purpose: Pull real country-year indicators from the World Bank API,
filter to real countries, and save raw + combined CSVs to data/raw/.
"""

import requests
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime, timedelta

RAW_DIR = "data/raw"
LAST_CHECKED_FILE = os.path.join(RAW_DIR, ".last_checked.json")
COUNTRY_CACHE_FILE = os.path.join(RAW_DIR, "country_list_cache.json")
COUNTRY_CACHE_MAX_AGE_DAYS = 180  # ~6 months — country classifications rarely change

INDICATORS = {
    "mortality_u5":         {"code": "SH.DYN.MORT",   "true_source": "UN IGME (via World Bank API)"},
    "nutrition_stunting":   {"code": "SH.STA.STNT.ZS", "true_source": "World Bank WDI"},
    "sanitation_basic":     {"code": "SH.STA.BASS.ZS", "true_source": "WHO/UNICEF JMP (via World Bank API)"},
    "gdp_per_capita":       {"code": "NY.GDP.PCAP.CD", "true_source": "World Bank WDI"},
    "literacy_rate":        {"code": "SE.ADT.LITR.ZS", "true_source": "UNESCO Institute for Statistics (via World Bank API)"},
    "population":           {"code": "SP.POP.TOTL",    "true_source": "UN Population Division / World Bank WDI"},
    "cbr":                  {"code": "SP.DYN.CBRT.IN", "true_source": "World Bank WDI (Crude Birth Rate)"},
}

YEAR_RANGE = "2000:2023"


def request_with_retry(url: str, params: dict, max_attempts: int = 3, timeout: int = 30) -> requests.Response:
    """
    Wrapper around requests.get with retry + exponential backoff.
    """
    last_exception = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type and "javascript" not in content_type:
                raise ValueError(f"Expected JSON, got Content-Type: {content_type}")
            return response
        except (requests.exceptions.RequestException, ValueError) as e:
            last_exception = e
            wait = 2 ** attempt  # 2s, 4s, 8s
            print(f"Attempt {attempt}/{max_attempts} failed ({e}). Retrying in {wait}s...")
            time.sleep(wait)
    raise last_exception


def get_real_country_codes(force_refresh: bool = False) -> set:
    """
    Fetch the full country list from the World Bank API, excluding aggregates.
    """
    if not force_refresh and os.path.exists(COUNTRY_CACHE_FILE):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(COUNTRY_CACHE_FILE))
        if cache_age < timedelta(days=COUNTRY_CACHE_MAX_AGE_DAYS):
            with open(COUNTRY_CACHE_FILE, "r") as f:
                return set(json.load(f))
        else:
            print("Country list cache is stale (>180 days old) — refreshing.")

    url = "https://api.worldbank.org/v2/country"
    params = {"format": "json", "per_page": 400}
    response = request_with_retry(url, params)
    payload = response.json()

    real_countries = {
        entry["id"] for entry in payload[1]
        if entry["region"]["value"] != "Aggregates"
    }

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(COUNTRY_CACHE_FILE, "w") as f:
        json.dump(sorted(real_countries), f)

    print(f"Fetched and cached {len(real_countries)} real country codes.")
    return real_countries


def fetch_worldbank_indicator(indicator_key: str) -> tuple[pd.DataFrame, str | None]:
    """
    Pull one indicator for all countries across YEAR_RANGE.
    """
    info = INDICATORS[indicator_key]
    url = f"https://api.worldbank.org/v2/country/all/indicator/{info['code']}"
    params = {"format": "json", "date": YEAR_RANGE, "per_page": 20000}

    response = request_with_retry(url, params)
    payload = response.json()

    if len(payload) < 2 or payload[1] is None:
        print(f"WARNING: no data returned for {indicator_key}")
        return pd.DataFrame(), None

    metadata = payload[0]
    if metadata.get("pages", 1) > 1:
        print(f"WARNING: {indicator_key} has {metadata['pages']} pages — verifying coverage.")

    lastupdated = metadata.get("lastupdated")

    records = payload[1]
    rows = []
    for r in records:
        if r["value"] is None:
            continue
        rows.append({
            "country_code": r["countryiso3code"],
            "country_name": r["country"]["value"],
            "year": int(r["date"]),
            "indicator_name": indicator_key,
            "value": r["value"],
            "source": info["true_source"],
            "retrieved_date": datetime.now().strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows), lastupdated


def filter_to_real_countries(df: pd.DataFrame, real_country_codes: set) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["country_code"].isin(real_country_codes)].reset_index(drop=True)


def save_raw(df: pd.DataFrame, name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(RAW_DIR, f"{name}_{date_str}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")
    return filepath


def fetch_indicator_data(indicator_key: str, real_country_codes: set):
    df, lastupdated = fetch_worldbank_indicator(indicator_key)
    return filter_to_real_countries(df, real_country_codes), lastupdated


def check_for_updates() -> bool:
    """
    Real update check: compares each indicator's `lastupdated` metadata timestamp.
    """
    last_checked = {}
    if os.path.exists(LAST_CHECKED_FILE):
        with open(LAST_CHECKED_FILE, "r") as f:
            last_checked = json.load(f)

    any_changed = False
    current_timestamps = {}

    for key, info in INDICATORS.items():
        url = f"https://api.worldbank.org/v2/country/USA/indicator/{info['code']}"
        try:
            response = request_with_retry(url, {"format": "json", "mrv": 1}, max_attempts=2)
            payload = response.json()
            lastupdated = payload[0].get("lastupdated")
            current_timestamps[key] = lastupdated

            previous = last_checked.get(key)
            if previous != lastupdated:
                print(f"{key}: changed ({previous} -> {lastupdated})")
                any_changed = True
        except requests.exceptions.RequestException as e:
            print(f"Could not check {key}: {e}")
            continue

    with open(LAST_CHECKED_FILE, "w") as f:
        json.dump(current_timestamps, f)

    return any_changed


def pull_all() -> None:
    real_country_codes = get_real_country_codes()

    all_frames = []
    for indicator_key in INDICATORS.keys():
        print(f"Fetching {indicator_key}...")
        try:
            df, lastupdated = fetch_indicator_data(indicator_key, real_country_codes)
            save_raw(df, indicator_key)
            all_frames.append(df)
        except requests.exceptions.RequestException as e:
            print(f"ERROR fetching {indicator_key} after retries: {e}")
            continue

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        save_raw(combined, "combined")


if __name__ == "__main__":
    if "--check-only" in sys.argv:
        if check_for_updates():
            print("Update detected via lastupdated metadata — pulling fresh data.")
            pull_all()
        else:
            print("No changes detected since last check — skipping pull.")
    else:
        pull_all()
    print("Module 1 Ingestion Complete. Raw data saved to data/raw/.")