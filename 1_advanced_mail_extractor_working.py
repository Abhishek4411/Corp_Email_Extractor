import os, re, glob, asyncio, logging, hashlib, unicodedata, json, time
from datetime import datetime
from typing import List, Dict, Optional, TypedDict, Set
from urllib.parse import urljoin, urlparse
import random

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
import pandas as pd
import xlsxwriter

try:
    import dns.resolver as dnsresolver
    HAVE_DNS = True
except ModuleNotFoundError:
    HAVE_DNS = False

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11436")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama4:scout")
CONCURRENT_WORKERS = int(os.getenv("CONCURRENT_WORKERS", 12))
HTTP_TIMEOUT = 20
MAX_RETRIES = 3
LOG_FILE = "extractor.log"

HEADERS = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
]

EXECUTIVE_TITLES = [
    "ceo", "chief executive officer", "president", "founder", "co-founder",
    "cto", "chief technology officer", "cio", "chief information officer",
    "cfo", "chief financial officer", "coo", "chief operating officer",
    "cmo", "chief marketing officer", "vp", "vice president", "director",
    "head of", "manager", "lead", "senior", "principal", "architect"
]

CONTACT_PATHS = [
    "", "/contact", "/contact-us", "/about", "/about-us", "/team", "/leadership",
    "/management", "/people", "/staff", "/careers", "/jobs", "/investors",
    "/press", "/media", "/support", "/help", "/sales", "/business"
]

SEARCH_ENGINES = [
    "https://html.duckduckgo.com/html/?q={}",
    "https://www.bing.com/search?q={}",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("extractor")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.I)
BAD_FRAGMENTS = {
    "noreply", "no-reply", "donotreply", "spam", "abuse", "example", "security",
    "postmaster", "webmaster", "admin", "root", "mailer-daemon", "bounce",
    "unsubscribe", "privacy", "legal", "compliance", "automated", "system",
    "notification", "alerts", "monitoring", "logs", "test", "demo", "sample"
}
DOMAIN_RE = re.compile(r"https?://([^/]+)/?", re.I)
NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b")
PHONE_RE = re.compile(r"[\+]?[1-9]?[0-9]{7,15}")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.1, num_ctx=8192)

def strip_accents(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")

def extract_emails(text: str) -> List[str]:
    if not text:
        return []
    candidates = {e.lower() for e in EMAIL_RE.findall(text)}
    filtered = []
    for email in candidates:
        if any(frag in email for frag in BAD_FRAGMENTS):
            continue
        if email.count('@') != 1:
            continue
        local, domain = email.split('@')
        if len(local) < 2 or len(domain) < 4:
            continue
        if not re.match(r'^[a-zA-Z0-9.-]+$', domain):
            continue
        if domain.endswith(('.jpg', '.png', '.gif', '.svg', '.pdf', '.doc', '.zip')):
            continue
        filtered.append(email)
    return sorted(filtered)

def extract_domains(text: str) -> List[str]:
    if not text:
        return []
    domains = DOMAIN_RE.findall(text)
    cleaned = []
    for domain in domains:
        domain = domain.lower().strip()
        if domain.startswith('www.'):
            domain = domain[4:]
        if '.' in domain and len(domain) > 4:
            cleaned.append(domain)
    return sorted(set(cleaned))

def extract_names(text: str) -> List[str]:
    if not text:
        return []
    names = NAME_RE.findall(text)
    return [name.strip() for name in names if len(name.split()) >= 2]

def mx_exists(domain: str) -> bool:
    if not HAVE_DNS:
        return True
    try:
        dnsresolver.resolve(domain, "MX", lifetime=5)
        return True
    except Exception:
        return False

def score_email(email: str, domains: List[str], company: str) -> float:
    if not email:
        return 0.0
    
    score = 0.0
    local, domain = email.split('@')
    
    if domain in domains:
        score += 0.4
    
    company_words = company.lower().split()
    if any(word in domain for word in company_words if len(word) > 3):
        score += 0.3
    
    preferred_prefixes = ["info", "contact", "hello", "sales", "business", "support"]
    if any(local.startswith(prefix) for prefix in preferred_prefixes):
        score += 0.2
    
    if mx_exists(domain):
        score += 0.2
    
    if len(local) > 15 or '.' in local:
        score -= 0.1
    
    return min(score, 1.0)

async def ask_llm(prompt: str, max_tokens: int = 500) -> str:
    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a precise business intelligence assistant. Provide concise, accurate responses."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        log.error(f"LLM error: {e}")
        return ""

class State(TypedDict, total=False):
    company: str
    row: Dict
    domains: List[str]
    emails: Set[str]
    executives: List[Dict]
    processed_urls: Set[str]
    confidence_scores: Dict[str, float]
    best_emails: List[Dict]
    error: str
    step: str

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
    for attempt in range(retries):
        try:
            headers = random.choice(HEADERS)
            async with session.get(url, headers=headers, timeout=ClientTimeout(total=HTTP_TIMEOUT)) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'text/html' in content_type or 'text/plain' in content_type:
                        text = await response.text()
                        log.info(f"Successfully fetched {url} (attempt {attempt + 1})")
                        return text
                else:
                    log.warning(f"HTTP {response.status} for {url}")
        except Exception as e:
            log.warning(f"Fetch attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(random.uniform(1, 3))
    return None

async def seed_extraction(state: State) -> State:
    log.info(f"[{state['company']}] Starting seed extraction")
    
    row_text = " ".join(str(v) for v in state["row"].values() if v)
    state["emails"] = set(extract_emails(row_text))
    state["domains"] = extract_domains(row_text)
    state["processed_urls"] = set()
    state["confidence_scores"] = {}
    
    log.info(f"[{state['company']}] Found {len(state['emails'])} seed emails, {len(state['domains'])} domains")
    state["step"] = "domain_expansion"
    return state

async def domain_expansion(state: State) -> State:
    log.info(f"[{state['company']}] Expanding domain search")
    
    if len(state.get("domains", [])) < 3:
        prompt = f"""For the company "{state['company']}", list 5 most likely official website domains.
        Format: domain1.com, domain2.com, domain3.com
        Only provide the domains, no explanations."""
        
        try:
            llm_response = await ask_llm(prompt)
            suggested_domains = [d.strip().lower().replace("www.", "") 
                               for d in llm_response.split(",") if d.strip()]
            state["domains"].extend(suggested_domains)
            state["domains"] = list(set(state["domains"]))
            log.info(f"[{state['company']}] LLM suggested {len(suggested_domains)} additional domains")
        except Exception as e:
            log.error(f"[{state['company']}] Domain expansion failed: {e}")
    
    state["step"] = "multi_search"
    return state

async def multi_search_engine(state: State) -> State:
    log.info(f"[{state['company']}] Multi-engine search starting")
    
    search_queries = [
        f'"{state["company"]}" contact email',
        f'"{state["company"]}" executive team email',
        f'"{state["company"]}" management contact',
        f'"{state["company"]}" business development email',
        f'"{state["company"]}" sales team contact'
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for query in search_queries:
            for engine_url in SEARCH_ENGINES:
                formatted_url = engine_url.format(query.replace(" ", "+"))
                tasks.append(fetch_with_retry(session, formatted_url))
        
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_urls = set()
        for result in search_results:
            if isinstance(result, str) and result:
                soup = BeautifulSoup(result, 'html.parser')
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if href.startswith('http') and any(domain in href for domain in state.get("domains", [])):
                        all_urls.add(href)
        
        state["processed_urls"].update(all_urls)
        log.info(f"[{state['company']}] Found {len(all_urls)} relevant URLs from search")
    
    state["step"] = "deep_crawl"
    return state

async def deep_crawl(state: State) -> State:
    log.info(f"[{state['company']}] Deep crawling websites")
    
    all_urls = set()
    for domain in state.get("domains", [])[:5]:
        for path in CONTACT_PATHS:
            all_urls.add(f"https://{domain}{path}")
            all_urls.add(f"http://{domain}{path}")
    
    all_urls.update(state.get("processed_urls", set()))
    
    async with aiohttp.ClientSession() as session:
        crawl_tasks = [fetch_with_retry(session, url) for url in list(all_urls)[:50]]
        crawl_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)
        
        for url, result in zip(all_urls, crawl_results):
            if isinstance(result, str) and result:
                emails = extract_emails(result)
                names = extract_names(result)
                
                state["emails"].update(emails)
                
                for email in emails:
                    score = score_email(email, state.get("domains", []), state["company"])
                    state["confidence_scores"][email] = score
                
                log.info(f"[{state['company']}] {url}: found {len(emails)} emails, {len(names)} names")
    
    log.info(f"[{state['company']}] Total emails after crawling: {len(state['emails'])}")
    state["step"] = "linkedin_mining"
    return state

async def linkedin_mining(state: State) -> State:
    log.info(f"[{state['company']}] LinkedIn mining")
    
    linkedin_queries = [
        f'site:linkedin.com "{state["company"]}" CEO email',
        f'site:linkedin.com "{state["company"]}" CTO email',
        f'site:linkedin.com "{state["company"]}" executive',
        f'site:linkedin.com/company/{state["company"].replace(" ", "-").lower()}'
    ]
    
    async with aiohttp.ClientSession() as session:
        for query in linkedin_queries:
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            result = await fetch_with_retry(session, search_url)
            
            if result:
                soup = BeautifulSoup(result, 'html.parser')
                links = [a.get('href') for a in soup.find_all('a', href=True) 
                        if 'linkedin.com' in a.get('href', '')]
                
                for link in links[:5]:
                    linkedin_page = await fetch_with_retry(session, link)
                    if linkedin_page:
                        emails = extract_emails(linkedin_page)
                        state["emails"].update(emails)
                        
                        for email in emails:
                            score = score_email(email, state.get("domains", []), state["company"])
                            state["confidence_scores"][email] = score + 0.1
    
    state["step"] = "executive_generation"
    return state

async def executive_generation(state: State) -> State:
    log.info(f"[{state['company']}] Generating executive contacts")
    
    if not state.get("domains"):
        state["step"] = "final_validation"
        return state
    
    primary_domain = state["domains"][0]
    
    name_prompt = f"""List 10 likely names of executives (CEO, CTO, CIO, VP) at "{state['company']}".
    Format: FirstName LastName, FirstName LastName
    Only provide names, no titles or explanations."""
    
    try:
        llm_names = await ask_llm(name_prompt)
        executive_names = [name.strip() for name in llm_names.split(",") if name.strip()]
        
        state["executives"] = []
        for name in executive_names[:10]:
            if len(name.split()) >= 2:
                parts = name.split()
                first, last = parts[0], parts[-1]
                
                email_variations = [
                    f"{first.lower()}.{last.lower()}@{primary_domain}",
                    f"{first[0].lower()}{last.lower()}@{primary_domain}",
                    f"{first.lower()}{last.lower()}@{primary_domain}",
                    f"{first.lower()}_{last.lower()}@{primary_domain}"
                ]
                
                for email in email_variations:
                    state["emails"].add(email)
                    state["confidence_scores"][email] = 0.3
                
                state["executives"].append({
                    "name": name,
                    "emails": email_variations
                })
        
        log.info(f"[{state['company']}] Generated {len(state['executives'])} executive profiles")
    except Exception as e:
        log.error(f"[{state['company']}] Executive generation failed: {e}")
    
    state["step"] = "final_validation"
    return state

async def final_validation(state: State) -> State:
    log.info(f"[{state['company']}] Final validation and ranking")
    
    scored_emails = []
    for email in state.get("emails", set()):
        score = state.get("confidence_scores", {}).get(email, 0.1)
        scored_emails.append({
            "email": email,
            "confidence": score,
            "mx_valid": mx_exists(email.split('@')[1])
        })
    
    scored_emails.sort(key=lambda x: (x["mx_valid"], x["confidence"]), reverse=True)
    state["best_emails"] = scored_emails[:20]
    
    log.info(f"[{state['company']}] Validation complete: {len(scored_emails)} total, {len([e for e in scored_emails if e['mx_valid']])} MX valid")
    state["step"] = "complete"
    return state

workflow = StateGraph(State)
workflow.add_node("seed", seed_extraction)
workflow.add_node("domain_expansion", domain_expansion)
workflow.add_node("multi_search", multi_search_engine)
workflow.add_node("deep_crawl", deep_crawl) 
workflow.add_node("linkedin_mining", linkedin_mining)
workflow.add_node("executive_generation", executive_generation)
workflow.add_node("final_validation", final_validation)

workflow.set_entry_point("seed")
workflow.add_edge("seed", "domain_expansion")
workflow.add_edge("domain_expansion", "multi_search")
workflow.add_edge("multi_search", "deep_crawl")
workflow.add_edge("deep_crawl", "linkedin_mining")
workflow.add_edge("linkedin_mining", "executive_generation")
workflow.add_edge("executive_generation", "final_validation")
workflow.add_edge("final_validation", END)

pipeline = workflow.compile()

async def process_company(row: Dict) -> Dict:
    company_name = (
        row.get("CompanyName") or 
        row.get("company") or 
        row.get("Company") or 
        row.get("Name") or
        str(list(row.values())[0] if row.values() else "Unknown")
    ).strip()
    
    log.info(f"Processing company: {company_name}")
    
    initial_state: State = {
        "company": company_name,
        "row": row,
        "emails": set(),
        "step": "seed"
    }
    
    try:
        result = await pipeline.ainvoke(initial_state)
        
        best_emails = result.get("best_emails", [])
        primary_email = best_emails[0]["email"] if best_emails else ""
        
        return {
            "company": company_name,
            "primary_email": primary_email,
            "all_emails": [e["email"] for e in best_emails[:5]],
            "confidence": best_emails[0]["confidence"] if best_emails else 0.0,
            "total_found": len(result.get("emails", set())),
            "executives": result.get("executives", []),
            "domains": result.get("domains", [])
        }
        
    except Exception as e:
        log.error(f"Error processing {company_name}: {e}")
        return {
            "company": company_name,
            "primary_email": "",
            "all_emails": [],
            "confidence": 0.0,
            "error": str(e),
            "total_found": 0,
            "executives": [],
            "domains": []
        }

async def main():
    start_time = time.time()
    
    files = glob.glob("*.xlsx") + glob.glob("*.csv")
    if not files:
        print("No input files found")
        return
    
    input_file = files[0]
    log.info(f"Processing file: {input_file}")
    
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    if "CompanyName" not in df.columns:
        df.rename(columns={df.columns[0]: "CompanyName"}, inplace=True)
    
    companies = [row.to_dict() for _, row in df.iterrows() 
                if str(row.get("CompanyName", "")).strip()]
    
    total_companies = len(companies)
    print(f"Processing {total_companies} companies with {CONCURRENT_WORKERS} workers")
    print(f"Model: {OLLAMA_MODEL}")
    
    semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)
    
    async def process_with_semaphore(company_row):
        async with semaphore:
            return await process_company(company_row)
    
    results = []
    completed = 0
    
    for future in asyncio.as_completed([process_with_semaphore(company) for company in companies]):
        try:
            result = await future
            results.append(result)
            completed += 1
            
            status = result["primary_email"] if result["primary_email"] else "❌"
            print(f"{completed}/{total_companies} | {result['company'][:30]:<30} -> {status}")
            
        except Exception as e:
            log.error(f"Task failed: {e}")
            completed += 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"Advanced_Contacts_{timestamp}.xlsx"
    
    workbook = xlsxwriter.Workbook(output_file)
    worksheet = workbook.add_worksheet("Contact_Intelligence")
    
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#1f4e79',
        'font_color': 'white',
        'border': 1
    })
    
    headers = [
        "No", "Company", "Primary_Email", "Confidence", "Alternative_Emails", 
        "Total_Found", "Domains", "Executive_Count", "Processing_Time"
    ]
    
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
        worksheet.set_column(col, col, 25)
    
    for idx, result in enumerate(results, 1):
        alt_emails = "; ".join(result.get("all_emails", [])[:3])
        domains = "; ".join(result.get("domains", [])[:3])
        exec_count = len(result.get("executives", []))
        
        worksheet.write_row(idx, 0, [
            idx,
            result["company"],
            result["primary_email"],
            f"{result['confidence']:.2f}",
            alt_emails,
            result.get("total_found", 0),
            domains,
            exec_count,
            f"{time.time() - start_time:.1f}s"
        ])
    
    worksheet.autofilter(0, 0, len(results), len(headers) - 1)
    worksheet.freeze_panes(1, 0)
    workbook.close()
    
    elapsed = time.time() - start_time
    successful = len([r for r in results if r["primary_email"]])
    
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Success rate: {successful}/{total_companies} ({successful/total_companies*100:.1f}%)")
    print(f"Results saved to: {output_file}")
    print(f"Detailed logs in: {LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())