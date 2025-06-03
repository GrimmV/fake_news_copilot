from playwright.sync_api import sync_playwright
import time
import re
import pandas as pd
import os
import json
from datetime import datetime

def scrape_statements(url, start_id, all_results, current_page, checkpointed_id):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # Select all <li> items under the listicle
            items = page.query_selector_all("ul.o-listicle__list > li.o-listicle__item")
            
            if len(items) == 0:
                print("Finished scraping")
                return None

            for i, item in enumerate(items):
                if i + (current_page-1) * 30 >= checkpointed_id:
                    try:
                        article = item.query_selector("article.m-statement")
                        if not article:
                            continue

                        # Name
                        name_el = article.query_selector(".m-statement__name")
                        name = name_el.inner_text().strip() if name_el else "N/A"
                        speaker_relative_url = name_el.get_attribute("href") if name_el else None

                        # Date / context line
                        date_el = article.query_selector(".m-statement__desc")
                        date = date_el.inner_text().strip() if date_el else "N/A"

                        # Statement
                        quote_el = article.query_selector(".m-statement__quote a")
                        quote = quote_el.inner_text().strip() if quote_el else "N/A"

                        # Profile picture (larger version)
                        img_el = article.query_selector(".m-statement__image img.c-image__original")
                        img_url = img_el.get_attribute("src") if img_el else "N/A"
                        
                        # Label
                        label_el = article.query_selector(".m-statement__meter div picture img")
                        label = label_el.get_attribute("alt") if label_el else "N/A"
                        
                        # Visit the speaker detail page in a new tab
                        speaker_description = "N/A"
                        speaker_history = {}
                        date, context = extract_date_and_context(date)
                        numeric_label = get_numeric_label(label)
                        if numeric_label is None:
                            continue
                        
                        if speaker_relative_url:
                            detail_url = f"https://www.politifact.com{speaker_relative_url}"
                            speaker_page = browser.new_page()
                            try:
                                speaker_page.goto(detail_url)
                                speaker_page.wait_for_load_state("networkidle", timeout=20000)
                                time.sleep(1.5)  # Slight wait for JS-rendered content if needed

                                # Try to find a description element
                                desc_el = speaker_page.query_selector(".m-pageheader__body p")
                                speaker_description = desc_el.inner_text().strip() if desc_el else "N/A"
                                
                                # Scorecard parsing
                                history_section = speaker_page.query_selector_all(".m-scorecard__item[data-scorecard-item]")
                                for entry in history_section:
                                    title_el = entry.query_selector("h4.m-scorecard__title")
                                    title = title_el.inner_text().strip().split("\n")[0] if title_el else None

                                    check_el = entry.query_selector("p.m-scorecard__checks a")
                                    checks = check_el.inner_text().strip() if check_el else "0 Checks"

                                    if title:
                                        speaker_history[convert_count_title(title)] = get_checks(checks)
                            except Exception as e:
                                print(f"Error fetching speaker details: {str(e)}")
                            finally:
                                speaker_page.close()

                        result = {
                            "id": start_id + i,
                            "label": int(numeric_label) if numeric_label is not None else None,
                            "name": name,
                            "date": date,
                            "context": context,
                            "statement": quote,
                            "avatar": img_url,
                            "speaker_description": speaker_description,
                            **speaker_history
                        }
                        
                        all_results.append(result)
                        # Save after each statement
                        save_checkpoint(all_results, current_page, year, start_id + i)
                        print(f"Saved statement {start_id + i}")
                        
                    except Exception as e:
                        print(f"Error processing statement: {str(e)}")
                        # Save current progress even if there's an error
                        save_checkpoint(all_results, current_page, year, start_id + i)
                        continue
                
            browser.close()
            return len(all_results)

        except Exception as e:
            print(f"Error in page scraping: {str(e)}")
            return None
        finally:
            browser.close()

def get_numeric_label(label):
    print(label)
    if label == "pants-fire":
        return 0
    elif label == "false":
        return 1
    elif label == "mostly-false" or label == "barely-true":
        return 2
    elif label == "half-true":
        return 3
    elif label == "mostly-true":
        return 4
    elif label == "true":
        return 5
    else:
        return None
    
def convert_count_title(title):
    if title == "Pants on Fire":
        return "pants_on_fire_counts"
    elif title == "False":
        return "false_counts"
    elif title == "Mostly False":
        return "mostly_false_counts"
    elif title == "Half True":
        return "half_true_counts"
    elif title == "Mostly True":
        return "mostly_true_counts"
    elif title == "True":
        return "true_counts"
    else:
        return None

def get_checks(text):
    num = int(text[0])
    return num

def extract_date_and_context(text):

    match = re.search(r"stated on (.*?) in (.*?):", text)

    if match:
        date = match.group(1).strip()
        context = match.group(2).strip()
    else:
        date = None
        context = None
    
    return date, context

def save_checkpoint(all_results, current_page, year, checkpointed_id):
    # Save the data
    df = pd.DataFrame(all_results)
    df.to_csv(f"politifact_data_{year}.csv", index=False)
    
    # Save the progress
    checkpoint = {
        "last_page": current_page,
        "year": year,
        "checkpointed_id": checkpointed_id,
        "timestamp": datetime.now().isoformat()
    }
    with open("checkpoint.json", "w") as f:
        json.dump(checkpoint, f)

def load_checkpoint(year):
    if os.path.exists(f"checkpoint_{year}.json"):
        with open(f"checkpoint_{year}.json", "r") as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    
    # set year
    year = 2024
    checkpoint = load_checkpoint(year)
    if checkpoint:
        print(f"Resuming from checkpoint: Page {checkpoint['last_page']} of year {checkpoint['year']}")
        all_results = pd.read_csv(f"politifact_data_{checkpoint['year']}.csv").to_dict('records')
        page = checkpoint['last_page']
        year = checkpoint['year']
        checkpointed_id = checkpoint['checkpointed_id']
    else:
        all_results = []
        page = 1
        checkpointed_id = 0

    while True:
        url = f"https://www.politifact.com/factchecks/list/?page={page}&pubdate={year}"
        print(f"Scraping page {page} of year {year}...")
        
        items_scraped = scrape_statements(url, (page - 1) * 30, all_results, current_page=page, checkpointed_id=checkpointed_id)
        if not items_scraped:  # If no items found or error occurred
            break
            
        page += 1
    
    print(f"Scraping completed. Total statements: {len(all_results)}")
