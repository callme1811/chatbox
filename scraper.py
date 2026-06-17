import os
import urllib.request
import requests
from bs4 import BeautifulSoup
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at: {DATA_DIR}")

def download_medical_exam_law():
    """Downloads the 2023 Medical Examination and Treatment Law PDF from chinhphu.vn"""
    url = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2023/02/15luat.signed.pdf"
    dest_path = os.path.join(DATA_DIR, "Luat_Kham_benh_chua_benh_2023.pdf")
    
    if os.path.exists(dest_path):
        print("Luat_Kham_benh_chua_benh_2023.pdf already exists.")
        return dest_path

    print(f"Downloading Luat Kham benh chua benh 2023 from {url}...")
    try:
        # Avoid SSL certificate verify failed error just in case
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error downloading medical exam law: {e}")
        return None

def scrape_wikisource_law(url, filename, law_name):
    """Scrapes law text from a Wikisource URL and saves it as a text file"""
    dest_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest_path):
        print(f"{filename} already exists.")
        return dest_path
        
    print(f"Scraping {law_name} from {url}...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, "html.parser")
        parser_output = soup.find(class_="mw-parser-output")
        
        if not parser_output:
            print(f"Could not find mw-parser-output on page {url}")
            return None
            
        # Extract text content cleanly
        lines = []
        for elem in parser_output.find_all(["p", "h2", "h3", "h4", "li"]):
            # Ignore navigation or metadata sections if possible
            if elem.get("class") and "toc" in elem.get("class"):
                continue
            text = elem.get_text().strip()
            if not text:
                continue
                
            # If it's a heading, format it nicely
            if elem.name in ["h2", "h3", "h4"]:
                lines.append(f"\n\n## {text}\n")
            else:
                lines.append(text)
                
        full_text = "\n".join(lines)
        # Clean up excessive newlines
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"Successfully scraped and saved {law_name} to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error scraping {law_name}: {e}")
        return None

def download_all_default_laws():
    ensure_data_dir()
    
    # 1. Luật Khám bệnh, chữa bệnh 2023 (PDF)
    download_medical_exam_law()
    
    # 2. Luật Dược 2016 (TXT via Wikisource)
    scrape_wikisource_law(
        "https://vi.wikisource.org/wiki/Lu%E1%BA%ADt_d%C6%B0%E1%BB%A3c_n%C6%B0%E1%BB%9Bc_C%E1%BB%99ng_h%C3%B2a_x%C3%A3_h%E1%BB%99i_ch%E1%BB%A7_ngh%C4%A9a_Vi%E1%BB%87t_Nam_2016",
        "Luat_Duoc_2016.txt",
        "Luat Duoc 2016"
    )
    
    # 3. Luật Bảo hiểm y tế 2008 (TXT via Wikisource)
    scrape_wikisource_law(
        "https://vi.wikisource.org/wiki/Lu%E1%BA%ADt_b%E1%BA%A3o_hi%E1%BB%83m_y_t%E1%BA%BF_n%C6%B0%E1%BB%9Bc_C%E1%BB%99ng_h%C3%B2a_x%C3%A3_h%E1%BB%99i_ch%E1%BB%A7_ngh%C4%A9a_Vi%E1%BB%87t_Nam_2008",
        "Luat_Bao_hiem_y_te_2008.txt",
        "Luat Bao hiem y te 2008"
    )

if __name__ == "__main__":
    download_all_default_laws()
