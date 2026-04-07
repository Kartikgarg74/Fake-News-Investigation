"""Manages claims database — selection, filtering, and ground truth lookup."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"

# Difficulty tier mapping from LIAR labels
DIFFICULTY_MAP = {
    "true": "easy",
    "false": "easy",
    "mostly-true": "medium",
    "barely-true": "medium",  # LIAR uses 'barely-true' instead of 'mostly-false'
    "half-true": "hard",
    "pants-fire": "hard",
}

# Budget per difficulty tier
BUDGET_MAP = {
    "easy": 10,
    "medium": 8,
    "hard": 6,
}

# Source categories available for investigation
SOURCE_CATEGORIES = [
    "government_data",
    "academic_papers",
    "news_articles",
    "fact_checks",
    "medical_journals",
    "statistical_reports",
    "international_organizations",
    "industry_reports",
    "image_analysis",  # Visual evidence from associated images
]


class ClaimManager:
    """Manages the claims database and evidence mappings."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DATA_DIR / "claims.db")
        self._ensure_db()
        self._migrate_db()

    def _ensure_db(self):
        """Create claims DB if it doesn't exist (uses built-in sample data)."""
        db = Path(self.db_path)
        if db.exists():
            return

        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                label TEXT NOT NULL,
                speaker TEXT DEFAULT '',
                topic TEXT DEFAULT '',
                difficulty TEXT NOT NULL,
                gold_evidence TEXT DEFAULT '[]',
                gold_reasoning TEXT DEFAULT '',
                evidence_passages TEXT DEFAULT '{}',
                image_url TEXT DEFAULT NULL
            )
        """)

        # Insert sample claims for each difficulty tier
        sample_claims = self._get_sample_claims()
        cur.executemany(
            """INSERT OR IGNORE INTO claims
               (id, claim, label, speaker, topic, difficulty,
                gold_evidence, gold_reasoning, evidence_passages, image_url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            sample_claims,
        )
        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Add new columns to existing databases (safe no-op if already present)."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("ALTER TABLE claims ADD COLUMN image_url TEXT DEFAULT NULL")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    def get_random_claim(self, difficulty: str = "easy") -> Dict[str, Any]:
        """Pick a random claim from the specified difficulty tier."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM claims WHERE difficulty = ? ORDER BY RANDOM() LIMIT 1",
                (difficulty,),
            )
            row = cur.fetchone()

        if not row:
            raise ValueError(f"No claims found for difficulty: {difficulty}")

        import json

        return {
            "id": row["id"],
            "claim": row["claim"],
            "label": row["label"],
            "speaker": row["speaker"],
            "topic": row["topic"],
            "difficulty": row["difficulty"],
            "gold_evidence": json.loads(row["gold_evidence"]),
            "gold_reasoning": row["gold_reasoning"],
            "evidence_passages": json.loads(row["evidence_passages"]),
            "image_url": row["image_url"],
        }

    def get_claim_count(self, difficulty: Optional[str] = None) -> int:
        """Get number of claims, optionally filtered by difficulty."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            if difficulty:
                cur.execute(
                    "SELECT COUNT(*) FROM claims WHERE difficulty = ?", (difficulty,)
                )
            else:
                cur.execute("SELECT COUNT(*) FROM claims")
            count = cur.fetchone()[0]
        return count

    def _get_sample_claims(self) -> List[tuple]:
        """Built-in sample claims for development and testing.

        In production, these are replaced by the full LIAR dataset
        via data/setup_data.py.
        """
        import json

        claims = [
            # === EASY: Clear TRUE/FALSE claims ===
            (
                "easy_001",
                "The Great Wall of China is visible from space with the naked eye.",
                "false",
                "Common myth",
                "science",
                "easy",
                json.dumps(["nasa_statement", "astronaut_testimonies"]),
                "Multiple astronauts have confirmed the Great Wall is not visible from low Earth orbit with the naked eye. NASA has officially stated this. The wall is narrow (about 15 feet wide) and blends with the natural landscape.",
                json.dumps({
                    "government_data": "NASA has officially stated that the Great Wall of China is not visible from space with the naked eye. Astronaut William Pogue initially thought he saw it but later determined he was looking at the Grand Canal near Beijing.",
                    "academic_papers": "A 2004 study published in the journal Science by Chinese astronaut Yang Liwei confirmed he could not see the Great Wall from orbit. The wall is only about 15 feet wide, far too narrow to be seen from 200+ miles above Earth.",
                    "fact_checks": "This claim has been debunked by multiple fact-checking organizations including Snopes and NASA. The Great Wall, while long (13,000+ miles), is too narrow to be distinguished from surrounding terrain at orbital altitude.",
                }),
            ),
            (
                "easy_002",
                "Humans use only 10 percent of their brains.",
                "false",
                "Popular culture",
                "science",
                "easy",
                json.dumps(["neuroscience_studies", "brain_imaging_research"]),
                "Brain imaging studies show that virtually all areas of the brain are active and have known functions. No area is completely inactive. This myth has been thoroughly debunked by neuroscience.",
                json.dumps({
                    "medical_journals": "Neuroimaging studies using fMRI and PET scans show that over the course of a day, virtually all brain areas are active. Even during sleep, many brain areas remain active. The 10% myth has no scientific basis.",
                    "academic_papers": "Research published in Nature Reviews Neuroscience demonstrates that every region of the brain has a known function. Brain damage to even small areas can have devastating effects, contradicting the 10% myth.",
                    "fact_checks": "The '10% of the brain' claim is one of the most persistent neuromyths. It has been debunked by the British Medical Journal, Scientific American, and numerous neuroscientists.",
                }),
            ),
            (
                "easy_003",
                "Water boils at 100 degrees Celsius at sea level.",
                "true",
                "Science textbook",
                "science",
                "easy",
                json.dumps(["physics_reference", "measurement_standards"]),
                "This is a well-established physical fact. Water boils at exactly 100°C (212°F) at standard atmospheric pressure (1 atm) at sea level. This is used as a calibration point for the Celsius temperature scale.",
                json.dumps({
                    "academic_papers": "The boiling point of water at standard atmospheric pressure (101.325 kPa) is 100°C. This was originally used to define the Celsius scale. At higher altitudes, water boils at lower temperatures due to reduced atmospheric pressure.",
                    "government_data": "NIST (National Institute of Standards and Technology) confirms the boiling point of pure water at 1 standard atmosphere is 99.97°C (very close to 100°C, the difference due to modern redefinition of the Celsius scale).",
                }),
            ),
            (
                "easy_004",
                "Lightning never strikes the same place twice.",
                "false",
                "Common saying",
                "science",
                "easy",
                json.dumps(["weather_data", "lightning_research"]),
                "Lightning frequently strikes the same place multiple times. The Empire State Building is struck about 20-25 times per year. Tall structures and certain geographical features attract repeated lightning strikes.",
                json.dumps({
                    "government_data": "The National Weather Service reports that the Empire State Building is struck by lightning approximately 20-25 times per year, directly disproving this myth.",
                    "academic_papers": "Lightning research shows that tall, pointed structures and certain terrain features act as preferred strike points. A single thunderstorm can strike the same location multiple times within minutes.",
                }),
            ),
            (
                "easy_005",
                "The Earth revolves around the Sun.",
                "true",
                "Science",
                "science",
                "easy",
                json.dumps(["astronomy_data", "space_observation"]),
                "The heliocentric model has been established since Copernicus (1543) and confirmed by centuries of astronomical observation. Earth orbits the Sun at an average distance of about 93 million miles.",
                json.dumps({
                    "academic_papers": "The heliocentric model was established by Copernicus in 1543, supported by Galileo's observations, and confirmed by Kepler's laws of planetary motion. Earth completes one orbit around the Sun every 365.25 days.",
                    "government_data": "NASA confirms Earth orbits the Sun at an average distance of 92.96 million miles (149.6 million km), completing one orbit in approximately 365.25 days.",
                }),
            ),

            # === MEDIUM: Distorted/exaggerated claims ===
            (
                "medium_001",
                "A Harvard study found that drinking coffee reduces cancer risk by 50 percent.",
                "barely-true",
                "Health blog",
                "health",
                "medium",
                json.dumps(["harvard_coffee_study", "cancer_meta_analyses"]),
                "Harvard researchers did find some association between coffee consumption and reduced risk of certain cancers, but the figure of 50% is a gross exaggeration. The actual findings showed approximately 15% reduction in risk of colorectal cancer specifically, not all cancers. The claim cherry-picks and inflates the actual research findings.",
                json.dumps({
                    "academic_papers": "A 2015 Harvard T.H. Chan School of Public Health study found that moderate coffee consumption (3-5 cups/day) was associated with reduced mortality from several causes. For specific cancers, the risk reduction was modest (10-20% for liver and colorectal cancer), not 50%.",
                    "medical_journals": "A comprehensive meta-analysis in the BMJ (2017) examining 201 studies found that coffee consumption was associated with a lower risk of several cancers, but the effect sizes were modest (relative risk reductions of 10-20%), far from the claimed 50%.",
                    "fact_checks": "Multiple fact-checkers have noted that health blogs frequently exaggerate study findings. The actual Harvard research showed modest protective effects for specific cancer types, not a 50% across-the-board reduction.",
                    "news_articles": "HealthNewsDaily.com reported 'Coffee Cuts Cancer Risk in Half!' — a sensationalized headline that misrepresents the actual study findings of modest risk reduction for specific cancer types.",
                }),
            ),
            (
                "medium_002",
                "Crime rates have doubled under the current administration.",
                "barely-true",
                "Political commentator",
                "crime",
                "medium",
                json.dumps(["fbi_crime_stats", "bjs_reports"]),
                "While certain specific crime categories (such as car theft or certain violent crimes in specific cities) may have increased significantly, the overall national crime rate has not doubled. The claim cherry-picks specific metrics while ignoring the broader trend. FBI UCR data shows a more nuanced picture.",
                json.dumps({
                    "government_data": "FBI Uniform Crime Report data shows that overall violent crime rates fluctuate year to year but have not doubled under any recent administration. Some categories increased while others decreased. Total property crime has generally trended downward over decades.",
                    "statistical_reports": "Bureau of Justice Statistics reports show that crime trends vary significantly by type and region. While some cities experienced spikes in certain categories, national aggregate data does not support a 'doubling' of crime rates.",
                    "news_articles": "Some media outlets highlighted increases in specific crime categories (e.g., motor vehicle theft up 10.9% in 2022) while omitting that other categories declined. This selective reporting supports misleading narratives.",
                    "fact_checks": "PolitiFact and FactCheck.org have repeatedly noted that claims about crime 'doubling' or 'skyrocketing' under various administrations are typically exaggerations that cherry-pick specific metrics or locations.",
                }),
            ),
            (
                "medium_003",
                "Renewable energy is now cheaper than fossil fuels in every country.",
                "barely-true",
                "Clean energy advocate",
                "energy",
                "medium",
                json.dumps(["irena_cost_reports", "iea_analysis"]),
                "While renewable energy costs have dropped dramatically and new solar/wind is cheaper than new coal/gas plants in most countries, the claim that this is true 'in every country' is an overstatement. Some regions still face higher renewable costs due to geography, infrastructure, and grid integration challenges.",
                json.dumps({
                    "international_organizations": "IRENA's 2023 Renewable Power Generation Costs report shows that 86% of new renewable capacity added in 2022 had lower costs than new fossil fuel plants. However, this is not 100% of countries — some developing nations face higher costs due to financing and infrastructure gaps.",
                    "statistical_reports": "The IEA World Energy Outlook notes that while utility-scale solar is the cheapest source of new electricity generation in most markets, grid integration costs and storage requirements can increase the effective cost in some regions.",
                    "industry_reports": "BloombergNEF reports that the levelized cost of electricity (LCOE) for solar and wind has fallen below fossil fuels in most countries, but intermittency and storage costs mean the full system cost comparison is more nuanced.",
                }),
            ),

            # === HARD: Sophisticated misinformation ===
            (
                "hard_001",
                "The economy grew 12 percent last quarter according to official government statistics.",
                "half-true",
                "Government official",
                "economy",
                "hard",
                json.dumps(["gdp_raw_data", "economic_context", "base_effect_analysis"]),
                "While the official GDP figure may technically show 12% growth, this is misleading due to the base effect — the previous year's quarter had an unusually low GDP due to pandemic lockdowns. The adjusted growth rate, removing the base effect, is approximately 2-3%. The claim is technically accurate but deeply misleading without context.",
                json.dumps({
                    "government_data": "Official GDP data shows 12% year-over-year growth. However, this comparison is against a quarter where GDP contracted sharply due to pandemic restrictions. Quarter-over-quarter growth (seasonally adjusted) was approximately 2.1%.",
                    "statistical_reports": "Economists note that year-over-year GDP comparisons during and after economic disruptions are misleading due to the 'base effect.' When comparing to the same quarter two years prior (pre-pandemic), growth is approximately 3.2%, not 12%.",
                    "academic_papers": "Research on GDP measurement methodology emphasizes that base effects can create misleading growth figures. The IMF recommends using seasonally adjusted quarter-over-quarter rates or comparing to pre-crisis baselines during recovery periods.",
                    "news_articles": "Several economic outlets reported the 12% figure prominently, while others noted the base effect. Some outlets with known biases presented the 12% figure without context to support political narratives.",
                    "international_organizations": "The World Bank's analysis of post-pandemic recoveries notes that many countries showed 'artificially high' growth rates due to base effects, cautioning against taking these figures at face value.",
                    "fact_checks": "Multiple fact-checkers rated similar claims as 'misleading' because the 12% figure, while technically from official data, omits critical context about why the comparison period had unusually low GDP.",
                }),
            ),
            (
                "hard_002",
                "More people died from the flu than from COVID-19 last year.",
                "pants-fire",
                "Social media post",
                "health",
                "hard",
                json.dumps(["cdc_death_data", "who_mortality_data", "methodology_comparison"]),
                "This claim is categorically false. COVID-19 deaths significantly exceeded flu deaths in every year since the pandemic began. The claim sometimes arises from comparing different time periods, using preliminary vs. final data, or confusing case counts with death counts. CDC and WHO data consistently show COVID-19 mortality far exceeding influenza mortality.",
                json.dumps({
                    "government_data": "CDC data shows COVID-19 was among the leading causes of death, with hundreds of thousands of deaths annually in the US alone. Seasonal influenza typically causes 12,000-52,000 deaths per year in the US. COVID-19 deaths exceeded this by a factor of 5-20x depending on the year.",
                    "medical_journals": "Studies published in The Lancet and JAMA confirm that COVID-19's infection fatality rate was significantly higher than seasonal influenza, and total deaths far exceeded flu deaths even accounting for differences in testing and reporting.",
                    "international_organizations": "WHO global mortality data shows COVID-19 caused over 6.9 million confirmed deaths worldwide (with excess mortality estimates much higher), while seasonal flu causes an estimated 290,000-650,000 deaths globally per year.",
                    "statistical_reports": "Some versions of this claim compare a bad flu season's estimate to a mild COVID period, or compare 'with COVID' vs 'from COVID' death categories. These methodological tricks distort the comparison.",
                    "fact_checks": "This claim has been rated 'Pants on Fire' by PolitiFact and 'False' by multiple fact-checking organizations. The data overwhelmingly shows COVID-19 deaths far exceeding flu deaths.",
                    "news_articles": "Some outlets with low factual reporting scores promoted this claim by cherry-picking time periods or using preliminary data that was later revised upward.",
                }),
            ),
            (
                "hard_003",
                "Electric vehicles produce more lifetime carbon emissions than gasoline cars when you account for battery manufacturing.",
                "half-true",
                "Auto industry analyst",
                "environment",
                "hard",
                json.dumps(["lifecycle_analyses", "battery_manufacturing_data", "grid_mix_studies"]),
                "This claim has a kernel of truth but is ultimately misleading. EV battery manufacturing does produce significant emissions, and in regions with very coal-heavy electricity grids, lifecycle emissions can approach those of efficient gasoline cars. However, in most regions and over the full vehicle lifetime, EVs produce significantly fewer total emissions. The claim cherry-picks the worst-case scenario and presents it as universal.",
                json.dumps({
                    "academic_papers": "A comprehensive lifecycle analysis in Nature Energy (2020) found that EVs produce 50-70% fewer lifecycle CO2 emissions than gasoline cars in most countries. However, in countries with very coal-heavy grids (e.g., Poland), the advantage is smaller or negligible.",
                    "government_data": "The US DOE and EPA analyses show that even accounting for battery manufacturing, EVs produce fewer emissions over their lifetime in all 50 US states, though the margin varies significantly by regional grid mix.",
                    "industry_reports": "The International Council on Clean Transportation (ICCT) found that battery manufacturing adds 30-40% more manufacturing emissions for EVs compared to gasoline cars, but this is offset within 1.5-3 years of driving depending on the electricity source.",
                    "statistical_reports": "Some analyses that show EVs as worse use outdated battery manufacturing data (pre-2018) when production was less efficient, or assume a 100% coal grid that doesn't exist in most countries.",
                    "international_organizations": "The IEA confirms that on a global average basis, EVs produce significantly fewer lifecycle emissions, but acknowledges regional variation based on electricity generation mix.",
                }),
            ),
            # === VISUAL: Multimedia/image-based claims ===
            (
                "visual_001",
                "This photograph shows flooding in New York City caused by Hurricane Sandy in 2012.",
                "false",
                "Social media",
                "disaster",
                "easy",
                json.dumps(["image_analysis", "fact_checks"]),
                "The photograph actually shows flooding in Bangkok, Thailand during the Great Flood of 2011 — a full year before Hurricane Sandy. The image has been widely misattributed on social media by cropping out location context and reposting with false captions.",
                json.dumps({
                    "image_analysis": (
                        "The image shows extreme urban flooding with vehicles fully submerged "
                        "and multi-story buildings surrounded by water. Reverse image search and "
                        "EXIF metadata analysis confirms this photograph was taken in Bangkok, "
                        "Thailand during the Great Flood of 2011, approximately one year before "
                        "Hurricane Sandy struck the US East Coast. Thai-language signage is "
                        "partially visible on storefronts in the background. This image has been "
                        "documented by Snopes, AFP Fact Check, and Reuters as a repeatedly "
                        "misattributed photograph."
                    ),
                    "fact_checks": (
                        "Snopes and AFP Fact Check have both documented this specific image as "
                        "originating from the 2011 Thailand floods, not Hurricane Sandy. "
                        "The photograph has been misattributed dozens of times across different "
                        "natural disasters, each time with a new false caption."
                    ),
                }),
                "https://upload.wikimedia.org/wikipedia/commons/a/a8/2011_Thailand_flooding_Nakhon_Ratchasima.jpg",
            ),
            (
                "visual_002",
                "This chart proves crime has skyrocketed 400% in the last two years.",
                "half-true",
                "Political commentator",
                "crime",
                "hard",
                json.dumps(["image_analysis", "statistical_reports", "government_data"]),
                "The chart uses a truncated Y-axis starting at 950 rather than 0, making a modest increase from 980 to 1,020 cases appear as a near-vertical spike. The underlying data is real — crime did increase — but the visual presentation is deliberately misleading. The actual percentage increase is approximately 4%, not 400%.",
                json.dumps({
                    "image_analysis": (
                        "The chart displays crime statistics for a major metropolitan area. "
                        "Critical observation: the Y-axis begins at 950, not at 0. This truncation "
                        "makes a change from approximately 980 to 1,020 incidents appear as a "
                        "dramatic near-vertical increase occupying the full chart height. When "
                        "the same data is plotted on a zero-baseline axis, the increase is visually "
                        "modest. The chart lacks a source attribution, confidence intervals, or "
                        "methodology notes. The '400%' figure in the claim does not correspond to "
                        "any value visible in the chart — it appears to be fabricated. The actual "
                        "change represented is approximately 4.1%."
                    ),
                    "statistical_reports": (
                        "Bureau of Justice Statistics data for the referenced period shows a "
                        "modest single-digit percentage increase in the crime category displayed. "
                        "Statisticians flag truncated Y-axes as a common technique for "
                        "exaggerating trends in data visualizations."
                    ),
                    "government_data": (
                        "FBI Uniform Crime Report data does not support a 400% increase in any "
                        "major crime category over a two-year period for any US jurisdiction. "
                        "The largest documented increases for any specific category in any city "
                        "were in the range of 20-40% during post-pandemic spikes."
                    ),
                }),
                None,  # No specific image URL — visual_002 uses a described hypothetical chart
            ),
            (
                "visual_003",
                "Leaked photo shows a government official signing a secret executive order banning protests.",
                "pants-fire",
                "Anonymous social media account",
                "politics",
                "hard",
                json.dumps(["image_analysis", "fact_checks", "news_articles"]),
                "The photograph is AI-generated. Multiple visual forensic indicators confirm this: inconsistent lighting on hands, illegible background text (a hallmark of generative AI), mismatched shadow directions, and unnatural skin texture artifacts. No credible news outlet reported such an executive order, and official government records contain no such document.",
                json.dumps({
                    "image_analysis": (
                        "Forensic visual analysis reveals multiple indicators consistent with "
                        "AI image generation: (1) The subject's hands show inconsistent lighting "
                        "that does not match the room's light source direction. (2) Text visible "
                        "on documents in the background is partially illegible and contains "
                        "nonsensical character sequences — a known artifact of diffusion models. "
                        "(3) Shadow directions for the subject and background objects are "
                        "inconsistent, suggesting composite generation. (4) Skin texture on the "
                        "subject's face shows unnatural smoothness with periodic repetitive "
                        "patterns typical of GAN or diffusion model outputs. (5) The document "
                        "being 'signed' has no legible text despite being in sharp focus. "
                        "Multiple AI detection tools (Hive Moderation, AI or Not) classify "
                        "this image as AI-generated with >95% confidence."
                    ),
                    "fact_checks": (
                        "PolitiFact, Snopes, and Reuters Fact Check have all investigated this "
                        "image. All three concluded it is AI-generated and no executive order "
                        "matching the claim exists in official government records. The account "
                        "that originally posted it has a history of sharing AI-generated "
                        "political content."
                    ),
                    "news_articles": (
                        "No major news outlet — including those critical of the official in "
                        "question — reported on any executive order banning protests. The absence "
                        "of coverage across the full spectrum of political media is strong "
                        "evidence that the event depicted did not occur."
                    ),
                }),
                None,
            ),
        ]
        # Ensure all tuples have image_url as 10th element (None for text-only claims)
        return [c if len(c) == 10 else c + (None,) for c in claims]

