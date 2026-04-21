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
NEW_TRIAL_API_URL = os.environ.get("NEW_TRIAL_API_URL", "") 

DATA_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_results.json")
ARCHIVE_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_archive.json")
CACHE_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_urls.json")
METADATA_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_metadata.json")
WATCHLIST_FILE = os.path.join(BASE_DIR, f"{OUTPUT_PREFIX}_watchlist.json")

# STAGE 1 FILTER: Broad keywords to detect if an article is clinical/medical
STAGE_ONE_KEYWORDS = [
    "trial", "study", "phase", "topline", "data", "results", "endpoint", 
    "patient", "fda", "readout", "nsclc", "lung cancer", "obesity", "weight", 
    "glp-1", "oncology", "tumor", "efficacy", "safety", "clinical"
]

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
# 3. AI SEMANTIC BATCH SEARCH (STAGE 2)
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def passes_stage_one(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in STAGE_ONE_KEYWORDS)

async def batch_analyze_with_gemini(session, text, trials_list, source_url, source_label, pub_date, all_events):
    if not GEMINI_API_KEY or len(text) < 50:
        return 0

    trials_prompt = "\n".join([f"- NCT ID: {t['nct_id']} | Known Drug: {t['drug_name']}" for t in trials_list])
    
    # We send Gemini the text + ALL trials for this company at once.
    prompt = f"""
    You are an expert biotech clinical trial analyst. 
    Read the following news excerpt:
    {text[:8000]}
    
    We are tracking these specific trials:
    {trials_prompt}
    
    Does the excerpt report news, data, or updates for ANY of the trials listed above? 
    (Note: The excerpt may use a secret code name or mechanism of action instead of the known drug name. Use your biotech knowledge to connect them if applicable).
    
    If NO trials match, output exactly an empty JSON array: []
    If YES, output a raw JSON array of dictionaries for each matched trial (NO markdown, NO backticks):
    [
      {{
        "nct_id": "The matched NCT ID from the list",
        "status": "Choose one: [Actual Data Reported, Expected Future Data, Trial Status Update]",
        "sentiment": "Choose one: [Positive, Negative, Neutral]",
        "notes": "Write a clean 1-2 sentence clinical summary of what happened."
      }}
    ]
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    
    events_added = 0
    try:
        async with session.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=20) as response:
            if response.status == 200:
                data = await response.json()
                result_text = data['candidates'][0]['content']['parts'][0]['text']
                matches = json.loads(result_text)
                
                if not pub_date: pub_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
                
                for match in matches:
                    matched_nct = match.get("nct_id", "")
                    
                    # Find the original trial data from our watchlist
                    original_trial = next((t for t in trials_list if t['nct_id'] == matched_nct), None)
                    if not original_trial: continue
                    
                    # Check for duplicates on dashboard
                    existing_event = next((e for e in all_events if e['date'] == pub_date and e['source'] == source_url and matched_nct in e['nct_id']), None)
                    
                    if existing_event:
                        continue
                    
                    event_dict = {
                        "date": pub_date,
                        "ticker": original_trial['ticker'],
                        "nct_id": original_trial['nct_id'],
                        "indication": original_trial['indication'],
                        "drug_name": original_trial['drug_name'],
                        "status": match.get("status", "Trial Status Update"),
                        "sentiment": match.get("sentiment", "Neutral"),
                        "notes": match.get("notes", "AI matched this article to this trial."), 
                        "source": source_url,
                        "source_type": source_label,
                        "date_added": datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d'),
                        "is_new": True
                    }
                    all_events.append(event_dict)
                    events_added += 1
    except Exception as e:
        pass 
        
    return events_added

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
                    if row.get('drug_name', '').strip().lower() == 'placebo': continue
                    ctg_flag = str(row.get('ctg_results_only', '')).strip().lower()
                    row['ctg_results_only'] = ctg_flag in ['yes', 'y', 'true', '1']
                    watchlist.append(row)
        return watchlist
    
    if not os.path.exists(WATCHLIST_FILE): return []
    with open(WATCHLIST_FILE, 'r') as f: return json.load(f)

# ==========================================
# 5. CORE WORKERS
# ==========================================
async def scan_for_new_competitor_trials(session, all_events, scraped_urls, watchlist):
    if not NEW_TRIAL_API_URL: return
    print(f"\n📡 Scanning ClinicalTrials.gov API for new competitor trials...")
    
    fetch_url = NEW_TRIAL_API_URL + "&pageSize=20" if "pageSize" not in NEW_TRIAL_API_URL else NEW_TRIAL_API_URL
    data = await fetch_json(session, fetch_url)
    if not data or 'studies' not in data: return
        
    existing_ncts = {trial['nct_id'].strip().upper() for trial in watchlist}
    events_found = 0
    
    for study in data.get('studies', []):
        protocol = study.get('protocolSection', {})
        ident_module = protocol.get('identificationModule', {})
        nct_id = ident_module.get('nctId', '').upper()
        
        if not nct_id or nct_id in existing_ncts: continue
        cache_key = f"NEW_RADAR_{nct_id}"
        if cache_key in scraped_urls: continue
            
        title = ident_module.get('briefTitle', 'Unknown Title')
        sponsor_name = protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name', 'Unknown Sponsor')
        interventions_module = protocol.get('armsInterventionsModule', {}).get('interventions', [])
        drugs = [i.get('name') for i in interventions_module if i.get('type') in ['DRUG', 'BIOLOGICAL']]
        drug_name = " + ".join(drugs) if drugs else "Unknown Drug"
        
        status_module = protocol.get('statusModule', {})
        overall_status = status_module.get('overallStatus', 'Unknown')
        first_posted = status_module.get('studyFirstPostDateStruct', {}).get('date', '')
        
        try: clean_date = parsedate_to_datetime(first_posted).strftime('%Y-%m-%d')
        except: clean_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d')
        
        notes = f"🚨 NEW COMPETITOR TRIAL FILED. Sponsor: {sponsor_name}. Status: {overall_status}. Title: {title}"
        event_dict = {
            "date": clean_date, "ticker": sponsor_name[:15] + "..", "nct_id": nct_id,
            "indication": OUTPUT_PREFIX.upper(), "drug_name": drug_name,
            "status": "Trial Status Update", "sentiment": "Neutral", "notes": notes,
            "source": f"https://clinicaltrials.gov/study/{nct_id}", "source_type": "ClinicalTrials.gov",
            "date_added": datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d'), "is_new": True
        }
        all_events.append(event_dict)
        scraped_urls.add(cache_key)
        events_found += 1
    print(f"      [!] Discovered {events_found} brand new competitor trials!")

async def check_pubmed_for_trial(session, trial, all_events, scraped_urls):
    nct_id = trial['nct_id']
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={nct_id}&retmode=json&retmax=3&sort=date&reldate=180&datetype=edat"
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
        
        raw_date = article.get('sortpubdate', '')
        pub_date = None
        if raw_date:
            try: pub_date = raw_date.split(' ')[0].replace('/', '-') 
            except: pass
        
        article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if article_url in scraped_urls: continue
        
        # Pass to Gemini Batch processor (even though it's just 1 trial, it fits the unified architecture)
        combined_text = f"Medical Journal in {article.get('fulljournalname', '')}. Title: {article.get('title', '')}. Evaluates {nct_id}."
        await batch_analyze_with_gemini(session, combined_text, [trial], article_url, "PubMed Journal", pub_date, all_events)
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
        
        previous_ctg_events = [e for e in all_events if e.get('source_type') == 'ClinicalTrials.gov' and nct_id in e.get('nct_id', '')]
        status_changed = True
        if previous_ctg_events:
            previous_ctg_events.sort(key=lambda x: x['date'], reverse=True)
            if f"Status: {overall_status}" in previous_ctg_events[0].get('notes', ''): status_changed = False
                
        if trial.get('ctg_results_only') and not results_posted and not status_changed: return

        notes = f"Official CTG Update. Status: {overall_status}. " + ("Results Data Available on CTG." if results_posted else "")
        if last_update:
            exact_date = last_update.split('T')[0]
            date_cache_key = f"CTG_{nct_id}_{exact_date}"
            if date_cache_key in scraped_urls: return
            
            event_dict = {
                "date": exact_date, "ticker": trial['ticker'], "nct_id": nct_id,
                "indication": trial['indication'], "drug_name": trial['drug_name'],
                "status": "Actual Data Reported" if results_posted else "Trial Status Update",
                "sentiment": "Neutral", "notes": notes, "source": f"https://clinicaltrials.gov/study/{nct_id}",
                "source_type": "ClinicalTrials.gov", "date_added": datetime.now(ZoneInfo("America/Los_Angeles")).strftime('%Y-%m-%d'), "is_new": True
            }
            if not any(e['date'] == event_dict['date'] and trial['nct_id'] in e['nct_id'] and e['status'] == event_dict['status'] for e in all_events):
                all_events.append(event_dict)
            scraped_urls.add(date_cache_key)

async def scan_ticker_sources(session, scan_ticker, cik, trials_list, all_events, scraped_urls):
    events_found = 0
    
    # 1. YAHOO FINANCE SCAN
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

                combined = f"{title} {desc}"
                # STAGE 1: Does it look medical/clinical?
                if passes_stage_one(combined):
                    # STAGE 2: Let Gemini figure out if it matches any of our specific trials!
                    added = await batch_analyze_with_gemini(session, combined, trials_list, link, "Yahoo Finance", pub_date, all_events)
                    if added > 0:
                        events_found += added
                        scraped_urls.add(link)
        except: pass

    # 2. SEC SCAN
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
                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{str(int(cik))}/{acc_no}/{filings['primaryDocument'][idx]}"
                        
                        if doc_url in scraped_urls: continue
                        
                        html_content = await fetch_text(session, doc_url, SEC_HEADERS, use_semaphore=True)
                        if html_content:
                            soup = BeautifulSoup(html_content, 'html.parser')
                            text_clean = soup.get_text(separator=' ').replace('\n', ' ')
                            
                            # STAGE 1: Does the SEC doc mention clinical updates?
                            if passes_stage_one(text_clean):
                                # STAGE 2: Send to Gemini to find matches
                                added = await batch_analyze_with_gemini(session, text_clean, trials_list, doc_url, f"SEC {form}", filing_date, all_events)
                                if added > 0:
                                    events_found += added
                                    scraped_urls.add(doc_url)

    if events_found > 0:
        print(f"      [!] SUCCESS: AI Semantic Search found {events_found} updates for {scan_ticker}.")
    else:
        print(f"      [✓] Checked {scan_ticker} (No matches).")

# ==========================================
# 6. MASTER ORCHESTRATOR
# ==========================================
async def run_pipeline():
    active_events = load_state(DATA_FILE, list)
    archived_events = load_state(ARCHIVE_FILE, list)
    all_events = active_events + archived_events
    scraped_urls = set(load_state(CACHE_FILE, list))

    for event in all_events: event['is_new'] = False

    print("\n" + "="*60)
    print(f"INITIATING NCT CLINICAL TRIAL SCRAPER ({OUTPUT_PREFIX.upper()}) [AI SEMANTIC MODE]")
    print("="*60 + "\n")

    async with aiohttp.ClientSession() as session:
        watchlist = await load_watchlist(session)
        if not watchlist: return

        await scan_for_new_competitor_trials(session, all_events, scraped_urls, watchlist)

        print("\nFetching SEC Master Directory...")
        sec_dir_url = "https://www.sec.gov/files/company_tickers.json"
        sec_dir_data = await fetch_json(session, sec_dir_url, SEC_HEADERS)
        sec_mapping = {item['ticker'].upper(): {"cik": str(item['cik_str']).zfill(10), "name": item['title']} for item in sec_dir_data.values()} if sec_dir_data else {}

        ticker_groups = {}
        for trial in watchlist:
            partner_tickers = [t.strip().upper() for t in re.split(r'[,/]', trial['ticker']) if t.strip()]
            for t in partner_tickers:
                if t not in ticker_groups: ticker_groups[t] = []
                ticker_groups[t].append(trial)

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
                        
                        # STAGE 1 for Global Wires: Is it medical AND does it mention a tracked company?
                        if passes_stage_one(combined):
                            for t, trials_list in ticker_groups.items():
                                company_name = sec_mapping.get(t, {}).get("name", t)
                                if t in combined or company_name.split(' ')[0] in combined:
                                    # STAGE 2
                                    added = await batch_analyze_with_gemini(session, combined, trials_list, link, source_name, clean_date, all_events)
                                    if added > 0: scraped_urls.add(link)
                except: pass
        print("   -> PR Wire scan complete.\n")

        print(f"STEP 2: Scanning CTG API, PubMed, SEC, and Yahoo for {len(watchlist)} monitored trials...")
        tasks = []
        for trial in watchlist:
            tasks.append(check_clinicaltrials_gov(session, trial, all_events, scraped_urls))
            tasks.append(check_pubmed_for_trial(session, trial, all_events, scraped_urls))
            
        for t, trials_list in ticker_groups.items():
            cik = sec_mapping.get(t, {}).get("cik")
            tasks.append(scan_ticker_sources(session, t, cik, trials_list, all_events, scraped_urls))
        
        batch_size = 50
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"\n--- Executing {len(tasks)} total tasks across {total_batches} batches ---")
        for i in range(0, len(tasks), batch_size):
            print(f"⏳ Processing Batch {(i // batch_size) + 1} of {total_batches}...")
            await asyncio.gather(*tasks[i:i + batch_size])
            await asyncio.sleep(1) 

    cutoff_date = (datetime.now(ZoneInfo("America/Los_Angeles")) - timedelta(days=120)).strftime('%Y-%m-%d')
    new_active = [e for e in all_events if e.get('date', '2000-01-01') >= cutoff_date]
    new_archive = [e for e in all_events if e.get('date', '2000-01-01') < cutoff_date]

    new_active.sort(key=lambda x: x['date'], reverse=True)
    new_archive.sort(key=lambda x: x['date'], reverse=True)
    
    save_state(new_active, DATA_FILE)
    save_state(new_archive, ARCHIVE_FILE)
    save_state(list(scraped_urls), CACHE_FILE)
    save_state({"last_updated": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%B %d, %Y at %I:%M %p")}, METADATA_FILE)
        
    print("\n" + "="*60)
    print(f"Run Complete! Exported {len(new_active)} Active and {len(new_archive)} Archived trial updates.")
    print("="*60)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
