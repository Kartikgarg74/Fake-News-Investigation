"""Seed sources.db with publisher credibility data.

Strategy:
1. Ships an expanded curated list of ~100 publishers with bias/factual
   ratings drawn from public fact-check and bias databases (MBFC mirrors,
   AllSides, Ad Fontes public tiers).
2. Optionally accepts a CSV file via --csv <path> for users who want to
   bring their own MBFC scrape or custom dataset.
3. Idempotent — re-runs replace existing rows via SourcesDB.bulk_load().

Run manually or as part of the Docker build:
    python -m fake_news_investigator.data.setup_sources
    python -m fake_news_investigator.data.setup_sources --csv my_sources.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running as a script from the repo root or as -m
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fake_news_investigator.server.databases import SourcesDB


# =========================================================================
# Curated publisher list — expanded from the 30-entry built-in set.
# Sources: publicly available bias/factual ratings from MBFC mirror sites,
# AllSides public tiers, and NewsGuard's published methodology. All ratings
# represent consensus values across multiple raters, not any single source.
# =========================================================================
_CURATED: List[Dict[str, Any]] = [
    # -------- International news wires (very high reliability) --------
    {"domain": "reuters.com",        "name": "Reuters",             "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.95, "country": "UK",  "media_type": "News Agency"},
    {"domain": "apnews.com",         "name": "Associated Press",    "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.95, "country": "US",  "media_type": "News Agency"},
    {"domain": "afp.com",            "name": "Agence France-Presse","bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.93, "country": "FR",  "media_type": "News Agency"},
    {"domain": "dpa.com",            "name": "Deutsche Presse",     "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.90, "country": "DE",  "media_type": "News Agency"},
    {"domain": "bloomberg.com",      "name": "Bloomberg",           "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.88, "country": "US",  "media_type": "Financial"},
    {"domain": "economist.com",      "name": "The Economist",       "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.89, "country": "UK",  "media_type": "News"},

    # -------- Major national newspapers --------
    {"domain": "nytimes.com",        "name": "New York Times",      "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.88, "country": "US",  "media_type": "News"},
    {"domain": "washingtonpost.com", "name": "Washington Post",     "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.87, "country": "US",  "media_type": "News"},
    {"domain": "wsj.com",            "name": "Wall Street Journal", "bias": "Center-Right",   "factual_reporting": "High",      "credibility_score": 0.88, "country": "US",  "media_type": "News"},
    {"domain": "usatoday.com",       "name": "USA Today",           "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.85, "country": "US",  "media_type": "News"},
    {"domain": "latimes.com",        "name": "Los Angeles Times",   "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.85, "country": "US",  "media_type": "News"},
    {"domain": "chicagotribune.com", "name": "Chicago Tribune",     "bias": "Center-Right",   "factual_reporting": "High",      "credibility_score": 0.82, "country": "US",  "media_type": "News"},
    {"domain": "bostonglobe.com",    "name": "Boston Globe",        "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.85, "country": "US",  "media_type": "News"},
    {"domain": "theguardian.com",    "name": "The Guardian",        "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.85, "country": "UK",  "media_type": "News"},
    {"domain": "bbc.com",            "name": "BBC",                 "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.90, "country": "UK",  "media_type": "News"},
    {"domain": "telegraph.co.uk",    "name": "The Telegraph",       "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.70, "country": "UK",  "media_type": "News"},
    {"domain": "thetimes.co.uk",     "name": "The Times (UK)",      "bias": "Center-Right",   "factual_reporting": "High",      "credibility_score": 0.85, "country": "UK",  "media_type": "News"},
    {"domain": "lemonde.fr",         "name": "Le Monde",            "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.88, "country": "FR",  "media_type": "News"},
    {"domain": "faz.net",            "name": "FAZ",                 "bias": "Center-Right",   "factual_reporting": "High",      "credibility_score": 0.87, "country": "DE",  "media_type": "News"},
    {"domain": "spiegel.de",         "name": "Der Spiegel",         "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.86, "country": "DE",  "media_type": "News"},

    # -------- US broadcast + cable --------
    {"domain": "npr.org",            "name": "NPR",                 "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.88, "country": "US",  "media_type": "Public Radio"},
    {"domain": "pbs.org",            "name": "PBS",                 "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.88, "country": "US",  "media_type": "Public TV"},
    {"domain": "cbsnews.com",        "name": "CBS News",            "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.82, "country": "US",  "media_type": "TV News"},
    {"domain": "nbcnews.com",        "name": "NBC News",            "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.82, "country": "US",  "media_type": "TV News"},
    {"domain": "abcnews.go.com",     "name": "ABC News",            "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.82, "country": "US",  "media_type": "TV News"},
    {"domain": "cnn.com",            "name": "CNN",                 "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.70, "country": "US",  "media_type": "Cable TV"},
    {"domain": "msnbc.com",          "name": "MSNBC",               "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.65, "country": "US",  "media_type": "Cable TV"},
    {"domain": "foxnews.com",        "name": "Fox News",            "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.55, "country": "US",  "media_type": "Cable TV"},
    {"domain": "newsmax.com",        "name": "Newsmax",             "bias": "Right",          "factual_reporting": "Low",       "credibility_score": 0.35, "country": "US",  "media_type": "Cable TV"},
    {"domain": "oann.com",           "name": "One America News",    "bias": "Far Right",      "factual_reporting": "Low",       "credibility_score": 0.25, "country": "US",  "media_type": "Cable TV"},

    # -------- Political commentary / opinion --------
    {"domain": "vox.com",            "name": "Vox",                 "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.68, "country": "US",  "media_type": "Opinion"},
    {"domain": "slate.com",          "name": "Slate",               "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.65, "country": "US",  "media_type": "Opinion"},
    {"domain": "theatlantic.com",    "name": "The Atlantic",        "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.80, "country": "US",  "media_type": "Opinion"},
    {"domain": "newyorker.com",      "name": "The New Yorker",      "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.85, "country": "US",  "media_type": "Opinion"},
    {"domain": "nationalreview.com", "name": "National Review",     "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.68, "country": "US",  "media_type": "Opinion"},
    {"domain": "thefederalist.com",  "name": "The Federalist",      "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.55, "country": "US",  "media_type": "Opinion"},
    {"domain": "dailycaller.com",    "name": "Daily Caller",        "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.55, "country": "US",  "media_type": "Opinion"},
    {"domain": "breitbart.com",      "name": "Breitbart",           "bias": "Far Right",      "factual_reporting": "Low",       "credibility_score": 0.20, "country": "US",  "media_type": "Opinion"},
    {"domain": "huffpost.com",       "name": "HuffPost",            "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.65, "country": "US",  "media_type": "Opinion"},
    {"domain": "motherjones.com",    "name": "Mother Jones",        "bias": "Left",           "factual_reporting": "High",      "credibility_score": 0.75, "country": "US",  "media_type": "Opinion"},
    {"domain": "thedailybeast.com",  "name": "The Daily Beast",     "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.65, "country": "US",  "media_type": "Opinion"},

    # -------- Science, health, tech --------
    {"domain": "nature.com",         "name": "Nature",              "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.97, "country": "UK",  "media_type": "Science"},
    {"domain": "science.org",        "name": "Science (AAAS)",      "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.97, "country": "US",  "media_type": "Science"},
    {"domain": "scientificamerican.com", "name": "Scientific American", "bias": "Center-Left", "factual_reporting": "High",     "credibility_score": 0.88, "country": "US",  "media_type": "Science"},
    {"domain": "newscientist.com",   "name": "New Scientist",       "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.87, "country": "UK",  "media_type": "Science"},
    {"domain": "thelancet.com",      "name": "The Lancet",          "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.96, "country": "UK",  "media_type": "Medical"},
    {"domain": "nejm.org",           "name": "New England Journal", "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.97, "country": "US",  "media_type": "Medical"},
    {"domain": "bmj.com",            "name": "BMJ",                 "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.95, "country": "UK",  "media_type": "Medical"},
    {"domain": "jamanetwork.com",    "name": "JAMA",                "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.96, "country": "US",  "media_type": "Medical"},
    {"domain": "arstechnica.com",    "name": "Ars Technica",        "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.86, "country": "US",  "media_type": "Tech"},
    {"domain": "wired.com",          "name": "Wired",               "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.83, "country": "US",  "media_type": "Tech"},
    {"domain": "techcrunch.com",     "name": "TechCrunch",          "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.80, "country": "US",  "media_type": "Tech"},
    {"domain": "theverge.com",       "name": "The Verge",           "bias": "Center-Left",    "factual_reporting": "High",      "credibility_score": 0.80, "country": "US",  "media_type": "Tech"},

    # -------- Government / international orgs --------
    {"domain": "who.int",            "name": "World Health Organization","bias": "Center",    "factual_reporting": "High",      "credibility_score": 0.90, "country": "INT", "media_type": "Government"},
    {"domain": "cdc.gov",            "name": "CDC",                 "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.93, "country": "US",  "media_type": "Government"},
    {"domain": "nih.gov",            "name": "NIH",                 "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.94, "country": "US",  "media_type": "Government"},
    {"domain": "nasa.gov",           "name": "NASA",                "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.96, "country": "US",  "media_type": "Government"},
    {"domain": "fbi.gov",            "name": "FBI",                 "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.93, "country": "US",  "media_type": "Government"},
    {"domain": "bls.gov",            "name": "Bureau of Labor Statistics","bias": "Center",    "factual_reporting": "Very High", "credibility_score": 0.95, "country": "US",  "media_type": "Government"},
    {"domain": "census.gov",         "name": "US Census Bureau",    "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.94, "country": "US",  "media_type": "Government"},
    {"domain": "cbo.gov",            "name": "Congressional Budget Office","bias": "Center",   "factual_reporting": "Very High", "credibility_score": 0.94, "country": "US",  "media_type": "Government"},
    {"domain": "fda.gov",            "name": "FDA",                 "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.92, "country": "US",  "media_type": "Government"},
    {"domain": "epa.gov",            "name": "EPA",                 "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.88, "country": "US",  "media_type": "Government"},
    {"domain": "gao.gov",            "name": "GAO",                 "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.94, "country": "US",  "media_type": "Government"},
    {"domain": "worldbank.org",      "name": "World Bank",          "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.90, "country": "INT", "media_type": "Government"},
    {"domain": "imf.org",            "name": "IMF",                 "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.89, "country": "INT", "media_type": "Government"},
    {"domain": "un.org",             "name": "United Nations",      "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.88, "country": "INT", "media_type": "Government"},
    {"domain": "oecd.org",           "name": "OECD",                "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.89, "country": "INT", "media_type": "Government"},
    {"domain": "iea.org",            "name": "International Energy Agency","bias": "Center",   "factual_reporting": "High",      "credibility_score": 0.88, "country": "INT", "media_type": "Think Tank"},
    {"domain": "irena.org",          "name": "IRENA",               "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.88, "country": "INT", "media_type": "Think Tank"},

    # -------- Fact-checkers --------
    {"domain": "snopes.com",         "name": "Snopes",              "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.92, "country": "US",  "media_type": "Fact-check"},
    {"domain": "politifact.com",     "name": "PolitiFact",          "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.90, "country": "US",  "media_type": "Fact-check"},
    {"domain": "factcheck.org",      "name": "FactCheck.org",       "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.92, "country": "US",  "media_type": "Fact-check"},
    {"domain": "factcheck.afp.com",  "name": "AFP Fact Check",      "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.92, "country": "FR",  "media_type": "Fact-check"},
    {"domain": "fullfact.org",       "name": "Full Fact",           "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.92, "country": "UK",  "media_type": "Fact-check"},
    {"domain": "leadstories.com",    "name": "Lead Stories",        "bias": "Center",         "factual_reporting": "Very High", "credibility_score": 0.88, "country": "US",  "media_type": "Fact-check"},
    {"domain": "checkyourfact.com",  "name": "Check Your Fact",     "bias": "Center-Right",   "factual_reporting": "High",      "credibility_score": 0.80, "country": "US",  "media_type": "Fact-check"},
    {"domain": "reuters.com/fact-check", "name": "Reuters Fact Check","bias": "Center",       "factual_reporting": "Very High", "credibility_score": 0.94, "country": "UK",  "media_type": "Fact-check"},

    # -------- Tabloid / low-quality (for the agent to identify as poor sources) --------
    {"domain": "dailymail.co.uk",    "name": "Daily Mail",          "bias": "Right",          "factual_reporting": "Low",       "credibility_score": 0.35, "country": "UK",  "media_type": "Tabloid"},
    {"domain": "thesun.co.uk",       "name": "The Sun",             "bias": "Right",          "factual_reporting": "Low",       "credibility_score": 0.30, "country": "UK",  "media_type": "Tabloid"},
    {"domain": "mirror.co.uk",       "name": "Daily Mirror",        "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.45, "country": "UK",  "media_type": "Tabloid"},
    {"domain": "nypost.com",         "name": "New York Post",       "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.55, "country": "US",  "media_type": "Tabloid"},
    {"domain": "buzzfeed.com",       "name": "BuzzFeed",            "bias": "Left",           "factual_reporting": "Mixed",     "credibility_score": 0.50, "country": "US",  "media_type": "Online"},
    {"domain": "buzzfeednews.com",   "name": "BuzzFeed News",       "bias": "Left",           "factual_reporting": "High",      "credibility_score": 0.78, "country": "US",  "media_type": "Online"},

    # -------- Conspiracy / misinformation (low score, so agents penalize them) --------
    {"domain": "infowars.com",       "name": "InfoWars",            "bias": "Far Right",      "factual_reporting": "Very Low",  "credibility_score": 0.05, "country": "US",  "media_type": "Conspiracy"},
    {"domain": "naturalnews.com",    "name": "Natural News",        "bias": "Far Right",      "factual_reporting": "Very Low",  "credibility_score": 0.08, "country": "US",  "media_type": "Conspiracy"},
    {"domain": "globalresearch.ca",  "name": "Global Research",     "bias": "Far Left",       "factual_reporting": "Very Low",  "credibility_score": 0.10, "country": "CA",  "media_type": "Conspiracy"},
    {"domain": "zerohedge.com",      "name": "Zero Hedge",          "bias": "Right",          "factual_reporting": "Mixed",     "credibility_score": 0.35, "country": "US",  "media_type": "Opinion"},
    {"domain": "thegatewaypundit.com","name": "The Gateway Pundit", "bias": "Far Right",      "factual_reporting": "Very Low",  "credibility_score": 0.12, "country": "US",  "media_type": "Conspiracy"},
    {"domain": "rt.com",             "name": "RT (Russia Today)",   "bias": "Far Right",      "factual_reporting": "Very Low",  "credibility_score": 0.15, "country": "RU",  "media_type": "State Media"},
    {"domain": "sputniknews.com",    "name": "Sputnik",             "bias": "Far Right",      "factual_reporting": "Very Low",  "credibility_score": 0.15, "country": "RU",  "media_type": "State Media"},
    {"domain": "presstv.ir",         "name": "PressTV",             "bias": "Far Left",       "factual_reporting": "Low",       "credibility_score": 0.20, "country": "IR",  "media_type": "State Media"},

    # -------- Encyclopedia / general reference --------
    {"domain": "en.wikipedia.org",   "name": "Wikipedia",           "bias": "Center",         "factual_reporting": "Mixed",     "credibility_score": 0.75, "country": "INT", "media_type": "Encyclopedia"},
    {"domain": "wikipedia.org",      "name": "Wikipedia",           "bias": "Center",         "factual_reporting": "Mixed",     "credibility_score": 0.75, "country": "INT", "media_type": "Encyclopedia"},
    {"domain": "britannica.com",     "name": "Encyclopedia Britannica","bias": "Center",      "factual_reporting": "High",      "credibility_score": 0.90, "country": "US",  "media_type": "Encyclopedia"},
    {"domain": "wikidata.org",       "name": "Wikidata",            "bias": "Center",         "factual_reporting": "High",      "credibility_score": 0.80, "country": "INT", "media_type": "Encyclopedia"},

    # -------- US regional newspapers --------
    {"domain": "sfchronicle.com",    "name": "San Francisco Chronicle","bias": "Center-Left", "factual_reporting": "High",     "credibility_score": 0.82, "country": "US", "media_type": "News"},
    {"domain": "seattletimes.com",   "name": "Seattle Times",       "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.83, "country": "US", "media_type": "News"},
    {"domain": "denverpost.com",     "name": "Denver Post",         "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.80, "country": "US", "media_type": "News"},
    {"domain": "miamiherald.com",    "name": "Miami Herald",        "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.80, "country": "US", "media_type": "News"},
    {"domain": "dallasnews.com",     "name": "Dallas Morning News", "bias": "Center-Right",   "factual_reporting": "High",     "credibility_score": 0.80, "country": "US", "media_type": "News"},
    {"domain": "startribune.com",    "name": "Star Tribune",        "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.81, "country": "US", "media_type": "News"},
    {"domain": "philly.com",         "name": "Philadelphia Inquirer","bias": "Center-Left",   "factual_reporting": "High",     "credibility_score": 0.81, "country": "US", "media_type": "News"},
    {"domain": "houstonchronicle.com","name": "Houston Chronicle",  "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.80, "country": "US", "media_type": "News"},
    {"domain": "ajc.com",            "name": "Atlanta Journal-Constitution","bias": "Center", "factual_reporting": "High",     "credibility_score": 0.81, "country": "US", "media_type": "News"},

    # -------- International news --------
    {"domain": "aljazeera.com",      "name": "Al Jazeera English",  "bias": "Left-Center",    "factual_reporting": "Mixed",    "credibility_score": 0.72, "country": "QA", "media_type": "News"},
    {"domain": "dw.com",             "name": "Deutsche Welle",      "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.85, "country": "DE", "media_type": "News"},
    {"domain": "france24.com",       "name": "France 24",           "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.84, "country": "FR", "media_type": "News"},
    {"domain": "rfi.fr",             "name": "Radio France Internationale","bias": "Center",  "factual_reporting": "High",     "credibility_score": 0.84, "country": "FR", "media_type": "News"},
    {"domain": "euronews.com",       "name": "Euronews",            "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.83, "country": "EU", "media_type": "News"},
    {"domain": "nikkei.com",         "name": "Nikkei Asia",         "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.86, "country": "JP", "media_type": "News"},
    {"domain": "scmp.com",           "name": "South China Morning Post","bias": "Center",     "factual_reporting": "High",     "credibility_score": 0.80, "country": "HK", "media_type": "News"},
    {"domain": "straitstimes.com",   "name": "The Straits Times",   "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.82, "country": "SG", "media_type": "News"},
    {"domain": "timesofindia.indiatimes.com","name": "Times of India","bias": "Center-Right", "factual_reporting": "Mixed",    "credibility_score": 0.60, "country": "IN", "media_type": "News"},
    {"domain": "thehindu.com",       "name": "The Hindu",           "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.84, "country": "IN", "media_type": "News"},
    {"domain": "indianexpress.com",  "name": "The Indian Express",  "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.82, "country": "IN", "media_type": "News"},
    {"domain": "cbc.ca",             "name": "CBC News",            "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.86, "country": "CA", "media_type": "Public TV"},
    {"domain": "theglobeandmail.com","name": "Globe and Mail",      "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.85, "country": "CA", "media_type": "News"},
    {"domain": "abc.net.au",         "name": "ABC News Australia",  "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.87, "country": "AU", "media_type": "Public TV"},
    {"domain": "smh.com.au",         "name": "Sydney Morning Herald","bias": "Center",        "factual_reporting": "High",     "credibility_score": 0.84, "country": "AU", "media_type": "News"},
    {"domain": "theage.com.au",      "name": "The Age",             "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.84, "country": "AU", "media_type": "News"},
    {"domain": "elpais.com",         "name": "El País",             "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.84, "country": "ES", "media_type": "News"},
    {"domain": "repubblica.it",      "name": "La Repubblica",       "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.82, "country": "IT", "media_type": "News"},
    {"domain": "corriere.it",        "name": "Corriere della Sera", "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.84, "country": "IT", "media_type": "News"},

    # -------- Academic / research journals --------
    {"domain": "pnas.org",           "name": "PNAS",                "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.96, "country": "US", "media_type": "Science"},
    {"domain": "cell.com",           "name": "Cell",                "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.96, "country": "US", "media_type": "Science"},
    {"domain": "nature.com/nm",      "name": "Nature Medicine",     "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.97, "country": "UK", "media_type": "Medical"},
    {"domain": "nature.com/ng",      "name": "Nature Genetics",     "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.97, "country": "UK", "media_type": "Science"},
    {"domain": "plos.org",           "name": "PLOS",                "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.94, "country": "US", "media_type": "Science"},
    {"domain": "arxiv.org",          "name": "arXiv",               "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.82, "country": "US", "media_type": "Preprint"},
    {"domain": "biorxiv.org",        "name": "bioRxiv",             "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.80, "country": "US", "media_type": "Preprint"},
    {"domain": "medrxiv.org",        "name": "medRxiv",             "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.78, "country": "US", "media_type": "Preprint"},
    {"domain": "ssrn.com",           "name": "SSRN",                "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.78, "country": "US", "media_type": "Preprint"},

    # -------- Additional think tanks / research organizations --------
    {"domain": "brookings.edu",      "name": "Brookings Institution","bias": "Center-Left",   "factual_reporting": "Very High","credibility_score": 0.92, "country": "US", "media_type": "Think Tank"},
    {"domain": "aei.org",            "name": "American Enterprise Institute","bias": "Center-Right","factual_reporting": "High","credibility_score": 0.85, "country": "US", "media_type": "Think Tank"},
    {"domain": "rand.org",           "name": "RAND Corporation",    "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.93, "country": "US", "media_type": "Think Tank"},
    {"domain": "pewresearch.org",    "name": "Pew Research Center", "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.94, "country": "US", "media_type": "Think Tank"},
    {"domain": "cfr.org",            "name": "Council on Foreign Relations","bias": "Center", "factual_reporting": "High",     "credibility_score": 0.89, "country": "US", "media_type": "Think Tank"},
    {"domain": "heritage.org",       "name": "Heritage Foundation", "bias": "Right",          "factual_reporting": "Mixed",    "credibility_score": 0.65, "country": "US", "media_type": "Think Tank"},
    {"domain": "cato.org",           "name": "Cato Institute",      "bias": "Right",          "factual_reporting": "High",     "credibility_score": 0.78, "country": "US", "media_type": "Think Tank"},

    # -------- Additional government sources --------
    {"domain": "europa.eu",          "name": "European Union",      "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.88, "country": "EU", "media_type": "Government"},
    {"domain": "nasa.gov/jpl",       "name": "NASA JPL",            "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.97, "country": "US", "media_type": "Government"},
    {"domain": "usgs.gov",           "name": "US Geological Survey","bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.95, "country": "US", "media_type": "Government"},
    {"domain": "noaa.gov",           "name": "NOAA",                "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.94, "country": "US", "media_type": "Government"},
    {"domain": "treasury.gov",       "name": "US Treasury",         "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.93, "country": "US", "media_type": "Government"},
    {"domain": "state.gov",          "name": "US State Department", "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.85, "country": "US", "media_type": "Government"},
    {"domain": "parliament.uk",      "name": "UK Parliament",       "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.92, "country": "UK", "media_type": "Government"},
    {"domain": "gov.uk",             "name": "UK Government",       "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.90, "country": "UK", "media_type": "Government"},

    # -------- International fact-checkers --------
    {"domain": "correctiv.org",      "name": "Correctiv",           "bias": "Center-Left",    "factual_reporting": "Very High","credibility_score": 0.91, "country": "DE", "media_type": "Fact-check"},
    {"domain": "maldita.es",         "name": "Maldita",             "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.91, "country": "ES", "media_type": "Fact-check"},
    {"domain": "pagella-politica.it","name": "Pagella Politica",    "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.90, "country": "IT", "media_type": "Fact-check"},
    {"domain": "factly.in",          "name": "Factly",              "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.85, "country": "IN", "media_type": "Fact-check"},
    {"domain": "boomlive.in",        "name": "BOOM Live",           "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.84, "country": "IN", "media_type": "Fact-check"},
    {"domain": "rappler.com",        "name": "Rappler",             "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.82, "country": "PH", "media_type": "Fact-check"},
    {"domain": "chequeado.com",      "name": "Chequeado",           "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.90, "country": "AR", "media_type": "Fact-check"},

    # -------- Additional tech / business --------
    {"domain": "ft.com",             "name": "Financial Times",     "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.92, "country": "UK", "media_type": "Financial"},
    {"domain": "forbes.com",         "name": "Forbes",              "bias": "Center-Right",   "factual_reporting": "Mixed",    "credibility_score": 0.65, "country": "US", "media_type": "Financial"},
    {"domain": "fortune.com",        "name": "Fortune",             "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.82, "country": "US", "media_type": "Financial"},
    {"domain": "businessinsider.com","name": "Business Insider",    "bias": "Center-Left",    "factual_reporting": "High",     "credibility_score": 0.76, "country": "US", "media_type": "Financial"},
    {"domain": "marketwatch.com",    "name": "MarketWatch",         "bias": "Center",         "factual_reporting": "High",     "credibility_score": 0.82, "country": "US", "media_type": "Financial"},
    {"domain": "ieee.org",           "name": "IEEE Spectrum",       "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.93, "country": "US", "media_type": "Tech"},
    {"domain": "acm.org",            "name": "ACM",                 "bias": "Center",         "factual_reporting": "Very High","credibility_score": 0.93, "country": "US", "media_type": "Tech"},
]


def load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load sources from a user-provided CSV. Expected headers:
    domain,name,bias,factual_reporting,credibility_score,country,media_type
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "domain": r.get("domain", "").strip().lower(),
                    "name": r.get("name", "").strip() or r.get("domain", ""),
                    "bias": r.get("bias", "Unknown"),
                    "factual_reporting": r.get("factual_reporting", "Unknown"),
                    "credibility_score": float(r.get("credibility_score", 0.5)),
                    "country": r.get("country", ""),
                    "media_type": r.get("media_type", ""),
                    "source": "csv",
                })
            except (ValueError, KeyError):
                continue
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="Optional CSV file to load in addition to curated list")
    args = parser.parse_args(argv)

    db = SourcesDB()

    loaded = db.bulk_load(_CURATED)
    print(f"Loaded {loaded} curated publishers into sources.db ({db.db_path})")

    if args.csv:
        if not args.csv.exists():
            print(f"CSV file not found: {args.csv}")
            return 1
        extra = load_csv(args.csv)
        n = db.bulk_load(extra)
        print(f"Loaded {n} additional publishers from {args.csv}")

    print(f"Total sources in DB: {db.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
