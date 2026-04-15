import asyncio
import aiohttp
import json
import re
import os
import csv
import io
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

# ==========================================
# 1. CONFIGURATION (CLOUD & PARALLEL READY)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "nct")
WATCHLIST_URL = os.environ.get("WATCHLIST_URL", "")

DATA_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_results.json")
ARCHIVE_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_archive.json")
CACHE_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_urls.json")
METADATA_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_metadata.json")
WATCHLIST_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_watchlist.json")

ACTUAL_DATA_KEYWORDS = [
    "topline results", "top-line results", "met primary endpoint", "did not meet",
    "demonstrated", "showed", "statistically significant", "interim analysis",
    "positive data", "failed", "safety profile", "first look", "results showed"
]

FUTURE_EXPECTATION_KEYWORDS = [
    "expected in", "on track for", "anticipates reporting", "will report",
    "readout expected", "data expected", "guidance", "projected", "upcoming results"
]

SENTIMENT_DICT = {
    "Positive": ["met primary endpoint", "statistically significant", "positive results", "favorable safety", "well-tolerated", "efficacy demonstrated", "outperformed", "positive data", "encouraging", "success"],
    "Negative": ["failed to meet", "clinical hold", "adverse events", "did not meet", "safety concerns", "discontinued", "terminated", "missed primary", "failed"]
}

SEC_HEADERS = {'User-Agent': 'Johnson Widjaja (jwliauw@gmail.com)'}
YAHOO_HEADERS = {'User-Agent': 'Mozilla/5.0'}

TARGET_SEC_FORMS = ['8-K', '6-K']
SEC_SEMAPHORE = asyncio.Semaphore(5)
PUBMED_SEMAPHORE = asyncio.Semaphore(2)

MASTER_RSS_FEEDS = [
    ("GlobeNewswire", "https://www.globenewswire.com/RssFeed/industry/8000-Health%20Care/feedTitle/GlobeNewswire%20-%20Health%20Care"),
    ("PR Newswire", "https://www.prnewswire.com/rss/health-care/biotechnology-latest-news.rss"),
    ("BusinessWire", "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeGVtXWw=="),
    ("Roche", "https://www.roche.com/investors/news.xml"),
    ("Novo Nordisk", "https://www.novonordisk.com/content/nncorp/global/en/news-and-media/news-and-ir-materials/news-details.rss.xml"),
    ("Eli Lilly", "https://investor.lilly.com/rss/news-releases.xml"),
    ("Pfizer", "https://investors.pfizer.com/rss/news-releases.xml"),
    ("AstraZeneca", "https://www.astrazeneca.com/media-centre/press-releases.rss"),
    ("Amgen", "https://investors.amgen.com/rss/news-releases.xml")
]

# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
def load_state(filename, default_type):
    if not os.path.exists(filename): return default_type()
    with open(filename, 'r') as f:
        try: return json.load(f)
        except: return default_type()

def save_state(data, filename):
    with open(filename, 'w') as f: json.dump(data, f, indent=4)

# ==========================================
# 3. NLP CLASSIFICATION & GEMINI AI LOGIC
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def classify_tense(text):
    text_lower = text.lower()
    actual_score = sum(1 for kw in ACTUAL_DATA_KEYWORDS if kw in text_lower)
    future_score = sum(1 for kw in FUTURE_EXPECTATION_KEYWORDS if kw in text_lower)
    if future_score > actual_score: return "Expected Future Data"
    elif actual_score > 0: return "Actual Data Reported"
    else: return "Trial Mention (Unspecified)"

def map_sentiment(text):
    text_lower = text.lower()
    pos_score = sum(1 for kw in SENTIMENT_DICT["Positive"] if kw in text_lower)
    neg_score = sum(1 for kw in SENTIMENT_DICT["Negative"] if kw in text_lower)
    if pos_score > neg_score: return "Positive"
    elif neg_score > pos_score: return "Negative"
    return "Neutral"

def extract_context(text, nct_id, drug_name):
    sentences = re.split(r'(?<=[.!?]) +', text)
    drug_list = [d.strip().lower() for d in re.split(r'\+|&|\band\b', drug_name) if d.strip()]
    
    context_chunks = []
    for sentence in sentences:
        s_lower = sentence.lower()
        if nct_id.lower() in s_lower or (drug_list and any(d in s_lower for d in drug_list)):
            context_chunks.append(sentence.strip().replace('\n', ' '))
    
    if context_chunks:
        full_context = " ".join(context_chunks)
        return full_context[:1500] 
    return "Mentioned in document. See source for details."

async def analyze_with_gemini(session, context_text, trial):
    if not GEMINI_API_KEY or len(context_text) < 15:
        return classify_tense(context_text), map_sentiment(context_text), context_text[:350] + "..."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    You are an expert biotech clinical trial analyst. 
    Read the following excerpt regarding NCT ID {trial['nct_id']} and the drug(s) {trial['drug_name']}.
    
    Determine 3 things based strictly on the text provided, and output ONLY a raw JSON dictionary (no markdown tags, no backticks).
    1. "status": Choose exactly one of ["Actual Data Reported", "Expected Future Data", "Trial Status Update"].
    2. "sentiment": Choose exactly one of ["Positive", "Negative", "Neutral"].
    3. "notes": Write a clean, 1-2 sentence clinical summary of the news. Do not use the word "excerpt".

    Excerpt to analyze:
    {context_text}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    
    try:
        async with session.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                result_text = data['candidates'][0]['content']['parts'][0]['text']
                parsed = json.loads(result_text)
                
                return parsed.get("status", "Trial Status Update"), parsed.get("sentiment", "Neutral"), parsed.get("notes", context_text[:350])
    except Exception as e:
        pass 
        
    return classify_tense(context_text), map_sentiment(context_text), context_text[:350] + "..."

# ==========================================
# 4. ASYNC FETCHERS & LOADERS
# ==========================================
async def fetch_text(session, url, headers, use_semaphore=False):
    try:
        if use_semaphore:
            async with SEC_SEMAPHORE:
                await asyncio.sleep(0.2)
                async with session.get(url, headers=headers, timeout=45) as response:
                    return await response.text() if response.status == 200 else ""
        else:
            async with session.get(url, headers=headers, timeout=30) as response:
                return await response.text() if response.status == 200 else ""
    except Exception: return ""

async def fetch_json(session, url, headers=None, use_semaphore=False):
    try:
        if use_semaphore:
            async with SEC_SEMAPHORE:
                await asyncio.sleep(0.2)
                async with session.get(url, headers=headers, timeout=30) as response:
                    return await response.json() if response.status == 200 else {}
        else:
            async with session.get(url, headers=headers, timeout=30) as response:
                return await response.json() if response.status == 200 else {}
    except Exception: return {}

async def fetch_pubmed_json(session, url):
    try:
        async with PUBMED_SEMAPHORE:
            await asyncio.sleep(0.5)
            async with session.get(url, headers=YAHOO_HEADERS, timeout=30) as response:
                return await response.json() if response.status == 200 else {}
    except Exception: return {}

async def load_watchlist(session):
    if WATCHLIST_URL:
        print("Downloading master watchlist from Google Sheets...")
        csv_data = await fetch_text(session, WATCHLIST_URL, YAHOO_HEADERS)
        watchlist = []
        if csv_data:
            reader = csv.DictReader(io.StringIO(csv_data))
            for row in reader:
                if row.get('ticker') and row.get('nct_id'):
                    if row.get('drug_name', '').strip().lower() == 'placebo':
                        continue
                    
                    ctg_flag = str(row.get('ctg_results_only', '')).strip().lower()
                    row['ctg_results_only'] = ctg_flag in ['yes', 'y', 'true', '1']
                    
                    watchlist.append(row)
        return watchlist
    
    if not os.path.exists(WATCHLIST_FILE): return []
    with open(WATCHLIST_FILE, 'r') as f: return json.load(f)

# ==========================================
# 5. CORE WORKERS (GROUPED BY TICKER)
# ==========================================
async def check_pubmed_for_trial(session, trial, all_events, scraped_urls):
    nct_id = trial['nct_id']
    
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={nct_id}&retmode=json&retmax=3&sort=date"
    search_data = await fetch_pubmed_json(session, search_url)
    
    if not search_data or 'esearchresult' not in search_data: return
    pmids = search_data['esearchresult'].get('idlist', [])
    if not pmids: return
    
    pmid_str = ",".join(pmids)
    summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid_str}&retmode=json"
    summary_data = await fetch_pubmed_json(session, summary_url)
    
    if not summary_data or 'result' not in summary_data: return
    
    for pmid in pmids:
        if pmid not in summary_data['result']: continue
        
        article = summary_data['result'][pmid]
        title = article.get('title', '')
        journal = article.get('fulljournalname', '')
        
        article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if article_url in scraped_urls: continue
        
        combined_text = f"Medical Journal Publication in {journal}. Title: {title}. This peer-reviewed paper presents findings for clinical trial {nct_id} evaluating the drug {trial['drug_name']}."
        
        await process_text_for_trial(session, combined_text, trial, article_url, "PubMed Journal", None, all_events)
        scraped_urls.add(article_url)

async def check_clinicaltrials_gov(session, trial, all_events, scraped_urls):
    nct_id = trial['nct_id']
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    
    data = await fetch_json(session, url)
    if data and 'protocolSection' in data:
        status_module = data['protocolSection'].get('statusModule', {})
        last_update = status_module.get('lastUpdateSubmitDate', '')
        overall_status = status_module.get('overallStatus', '')
        
        results_posted = 'resultsSection' in data
        
        if trial.get('ctg_results_only') and not results_posted:
            return

        status_label = "Actual Data Reported" if results_posted else "Trial Status Update"
        notes = f"Official CTG Update. Status: {overall_status}. " + ("Results Data Available on CTG." if results_posted else "")
        sentiment = map_sentiment(notes)

        if last_update:
            exact_date = last_update.split('T')[0]
            date_cache_key = f"CTG_{nct_id}_{exact_date}"
            if date_cache_key in scraped_urls: return
            
            event_dict = {
                "date": exact_date,
                "ticker": trial['ticker'],
                "nct_id": nct_id,
                "indication": trial['indication'],
                "drug_name": trial['drug_name'],
                "status": status_label,
                "sentiment": sentiment,
                "notes": notes,
                "source": f"https://clinicaltrials.gov/study/{nct_id}",
                "source_type": "ClinicalTrials.gov",
                "date_added": datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d'),
                "is_new": True
            }
            
            is_duplicate = any(e['date'] == event_dict['date'] and trial['nct_id'] in e['nct_id'] and e['status'] == event_dict['status'] for e in all_events)
            if not is_duplicate:
                all_events.append(event_dict)
            
            scraped_urls.add(date_cache_key)

async def process_text_for_trial(session, text, trial, source_url, source_label, pub_date, all_events):
    events_added = 0
    text_lower = text.lower()
    
    drug_list = [d.strip().lower() for d in re.split(r'\+|&|\band\b', trial['drug_name']) if d.strip()]
    nct_mentioned = trial['nct_id'].lower() in text_lower
    drugs_mentioned = all(drug in text_lower for drug in drug_list) if drug_list else False

    if nct_mentioned or drugs_mentioned:
        raw_context = extract_context(text, trial['nct_id'], trial['drug_name'])
        status, sentiment, notes = await analyze_with_gemini(session, raw_context, trial)
        
        if not pub_date: pub_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
        
        existing_event = next((e for e in all_events if e['date'] == pub_date and e['source'] == source_url and e['ticker'] == trial['ticker'] and e['drug_name'] == trial['drug_name']), None)
        
        if existing_event:
            if trial['nct_id'] not in existing_event['nct_id']:
                existing_event['nct_id'] += f", {trial['nct_id']}"
            if trial['indication'] not in existing_event['indication']:
                existing_event['indication'] += f", {trial['indication']}"
            events_added += 1
        else:
            event_dict = {
                "date": pub_date,
                "ticker": trial['ticker'],
                "nct_id": trial['nct_id'],
                "indication": trial['indication'],
                "drug_name": trial['drug_name'],
                "status": status,
                "sentiment": sentiment,
                "notes": notes, 
                "source": source_url,
                "source_type": source_label,
                "date_added": datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d'),
                "is_new": True
            }
            all_events.append(event_dict)
            events_added += 1
            
    return events_added

async def scan_ticker_sources(session, scan_ticker, cik, trials_list, all_events, scraped_urls):
    events_found = 0
    
    yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={scan_ticker}&region=US&lang=en-US"
    rss_xml = await fetch_text(session, yahoo_url, YAHOO_HEADERS)
    if rss_xml:
        try:
            root = ET.fromstring(rss_xml)
            for item in root.findall('./channel/item'):
                link = item.find('link').text or ''
                if not link or link in scraped_urls: continue
                
                title = item.find('title').text or ''
                desc = item.find('description').text or ''
                pub_date = item.find('pubDate').text or ''
                try: pub_date = parsedate_to_datetime(pub_date).strftime('%Y-%m-%d')
                except: pub_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')

                added_for_doc = False
                for trial in trials_list:
                    added = await process_text_for_trial(session, f"{title} {desc}", trial, link, "Yahoo Finance", pub_date, all_events)
                    if added:
                        events_found += added
                        added_for_doc = True
                
                if added_for_doc: scraped_urls.add(link)
        except: pass

    if cik:
        sec_api_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = await fetch_json(session, sec_api_url, SEC_HEADERS, use_semaphore=True)
        if data and 'filings' in data:
            filings = data['filings']['recent']
            for idx, form in enumerate(filings['form']):
                if form in TARGET_SEC_FORMS:
                    filing_date = filings['filingDate'][idx]
                    if filing_date.startswith("2025") or filing_date.startswith("2026"):
                        acc_no = filings['accessionNumber'][idx].replace('-', '')
                        doc = filings['primaryDocument'][idx]
                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{str(int(cik))}/{acc_no}/{doc}"
                        
                        if doc_url in scraped_urls: continue
                        
                        html_content = await fetch_text(session, doc_url, SEC_HEADERS, use_semaphore=True)
                        if html_content:
                            soup = BeautifulSoup(html_content, 'html.parser')
                            text_clean = soup.get_text(separator=' ').replace('\n', ' ')
                            added_for_doc = False
                            for trial in trials_list:
                                added = await process_text_for_trial(session, text_clean, trial, doc_url, f"SEC {form}", filing_date, all_events)
                                if added:
                                    events_found += added
                                    added_for_doc = True
                            if added_for_doc: scraped_urls.add(doc_url)

    if events_found > 0:
        print(f"      [!] SUCCESS: Extracted {events_found} NCT updates via {scan_ticker}.")
    else:
        print(f"      [✓] Checked {scan_ticker} (No new trial updates).")

# ==========================================
# 6. MASTER ORCHESTRATOR
# ==========================================
async def run_pipeline():
    # MEMORY UPGRADE: Load both Active AND Archive files so the bot never forgets old URLs
    active_events = load_state(DATA_FILE, list)
    archived_events = load_state(ARCHIVE_FILE, list)
    all_events = active_events + archived_events
    scraped_urls = set(load_state(CACHE_FILE, list))

    for event in all_events: event['is_new'] = False

    print("\n" + "="*60)
    print(f"INITIATING NCT CLINICAL TRIAL SCRAPER ({OUTPUT_PREFIX.upper()})")
    print("="*60 + "\n")

    async with aiohttp.ClientSession() as session:
        watchlist = await load_watchlist(session)
        if not watchlist:
            print("No watchlist data found! Check your Google Sheet link.")
            return

        print("Fetching SEC Master Directory...")
        sec_dir_url = "https://www.sec.gov/files/company_tickers.json"
        sec_dir_data = await fetch_json(session, sec_dir_url, SEC_HEADERS)
        sec_mapping = {item['ticker'].upper(): str(item['cik_str']).zfill(10) for item in sec_dir_data.values()} if sec_dir_data else {}

        print("\nSTEP 1: Scanning Global PR Wires & Corporate Feeds...")
        for source_name, url in MASTER_RSS_FEEDS:
            xml_data = await fetch_text(session, url, YAHOO_HEADERS)
            if xml_data:
                try:
                    root = ET.fromstring(xml_data)
                    for item in root.findall('./channel/item'):
                        link = item.find('link').text or ''
                        if not link or link in scraped_urls: continue
                        
                        title = item.find('title').text or ''
                        desc = item.find('description').text or ''
                        pub_date = item.find('pubDate').text or ''
                        
                        try: clean_date = parsedate_to_datetime(pub_date).strftime('%Y-%m-%d')
                        except: clean_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')

                        combined = f"{title} {desc}"
                        
                        for trial in watchlist:
                            added = await process_text_for_trial(session, combined, trial, link, source_name, clean_date, all_events)
                            if added: scraped_urls.add(link)
                except: pass
        print("   -> PR Wire scan complete.\n")

        print(f"STEP 2: Scanning CTG API, PubMed, SEC, and Yahoo for {len(watchlist)} monitored trials...")
        tasks = []
        
        for trial in watchlist:
            tasks.append(check_clinicaltrials_gov(session, trial, all_events, scraped_urls))
            tasks.append(check_pubmed_for_trial(session, trial, all_events, scraped_urls))
        
        ticker_groups = {}
        for trial in watchlist:
            partner_tickers = [t.strip().upper() for t in re.split(r'[,/]', trial['ticker']) if t.strip()]
            for t in partner_tickers:
                if t not in ticker_groups:
                    ticker_groups[t] = []
                ticker_groups[t].append(trial)
                
        for t, trials_list in ticker_groups.items():
            cik = sec_mapping.get(t)
            tasks.append(scan_ticker_sources(session, t, cik, trials_list, all_events, scraped_urls))
        
        batch_size = 50
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n--- Executing {len(tasks)} total tasks across {total_batches} batches ---")
        
        for i in range(0, len(tasks), batch_size):
            current_batch = (i // batch_size) + 1
            print(f"⏳ Processing Batch {current_batch} of {total_batches}...")
            
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)
            await asyncio.sleep(1) 

    # --- NEW: AUTO-ARCHIVING SYSTEM (120 Days / 4 Months) ---
    cutoff_date = (datetime.now(ZoneInfo("America/Los_Angeles")) - timedelta(days=120)).strftime('%Y-%m-%d')
    
    new_active = []
    new_archive = []
    
    for e in all_events:
        evt_date = e.get('date', '2000-01-01')
        if evt_date >= cutoff_date:
            new_active.append(e)
        else:
            new_archive.append(e)

    # Sort both files by date
    new_active.sort(key=lambda x: x['date'], reverse=True)
    new_archive.sort(key=lambda x: x['date'], reverse=True)
    
    save_state(new_active, DATA_FILE)
    save_state(new_archive, ARCHIVE_FILE)
    save_state(list(scraped_urls), CACHE_FILE)
        
    run_time = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%B %d, %Y at %I:%M %p")
    save_state({"last_updated": run_time}, METADATA_FILE)
        
    print("\n" + "="*60)
    print(f"Run Complete! Exported {len(new_active)} Active and {len(new_archive)} Archived trial updates.")
    print("="*60)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
