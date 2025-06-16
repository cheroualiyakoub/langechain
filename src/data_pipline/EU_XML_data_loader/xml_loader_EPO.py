import xml.etree.ElementTree as ET
import re
import json
import os
from collections import defaultdict

def extract_heading_text(elem):
    """Extract text from heading element handling nested formatting tags"""
    # Try to convert the element to a string and use regex
    elem_str = ET.tostring(elem, encoding='unicode')
    
    # Try various patterns to extract the text
    for pattern in [r'<[bu]><[bu]>(.*?)</[bu]></[bu]>', r'<[bu]>(.*?)</[bu]>']:
        match = re.search(pattern, elem_str)
        if match:
            return match.group(1).strip()
    
    # If regex didn't work, try direct extraction
    if len(elem) > 0:
        if elem[0].tag in ['u', 'b', 'i']:
            if len(elem[0]) > 0 and elem[0][0].tag in ['u', 'b', 'i']:
                return (elem[0][0].text or "").strip()
            else:
                return (elem[0].text or "").strip()
        else:
            return (elem[0].text or "").strip()
    else:
        return (elem.text or "").strip()

def is_main_heading(heading_text, debug=False):
    """
    Stricter determination if a heading is a main section heading (not a compound or example).
    
    Args:
        heading_text: The heading text to analyze
        debug: Whether to print debug info
        
    Returns:
        bool: True if it's a main heading, False otherwise
    """
    # Strip and clean the heading text
    if not heading_text:
        return False
        
    # Convert to lowercase for case-insensitive matching
    heading_lower = heading_text.lower()
    
    # Explicit whitelist of main heading patterns (more precise than before)
    main_heading_whitelist = [
        r'^technical field$',
        r'^technical background$',
        r'^field of the invention$',
        r'^background$', 
        r'^background of the invention$',
        r'^summary$',
        r'^summary of the invention$',
        r'^brief summary$',
        r'^introduction$',
        r'^brief description of the drawings$',
        r'^detailed description$',
        r'^detailed description of the invention$',
        r'^abstract$',
        r'^description of embodiments$',
        r'^description of the embodiments$',
        r'^description of preferred embodiments$',
        r'^advantages$',
        r'^advantages of the invention$',
        r'^industrial applicability$',
        r'^brief description$',
        r'^objects of the invention$',
        r'^disclosure of the invention$',
        r'^action and effect of the invention$',
        r'^action and effect$'
    ]
    
    # Check exact matches for main headings first
    for pattern in main_heading_whitelist:
        if re.match(pattern, heading_lower):
            if debug:
                print(f"MAIN HEADING (whitelist): {heading_text}")
            return True
    
    # Extended patterns that require more context for verification
    extended_main_patterns = [
        r'^(figure|figures|drawings)',
        r'^embodiment',
        r'^description of',
        r'of the( present)? invention$',
    ]
    
    # Check extended patterns if no exact match was found
    for pattern in extended_main_patterns:
        if re.search(pattern, heading_lower):
            # Further verify it's not an example or compound
            if not re.search(r'example|compound|step|acid|\d+\s*[a-zA-Z]|\([0-9][a-zA-Z]+\)', heading_lower):
                if debug:
                    print(f"MAIN HEADING (extended): {heading_text}")
                return True
    
    # Extensive blacklist to catch all examples, compounds, and specific structures
    blacklist_patterns = [
        # Examples
        r'example\s*\d+',
        r'examples?\s*\d*[^$]',
        r'experimental example',
        r'comparative example',
        r'reference example',
        r'preparative example',
        
        # Compounds, chemicals, and formulas
        r'compound',
        r'chemical',
        r'formula',
        r'synthesis',
        r'preparation of',
        r'structure',
        r'moiety',
        r'\b[a-z]*acid',
        r'phosphoryl',
        r'benzamido',
        r'phenyl',
        r'methyl',
        r'ethyl',
        
        # Steps and procedures
        r'step [a-z0-9]',
        r'procedure',
        r'stage',
        
        # Statistical patterns
        r'\d+\s*\.?\s*\d*\s*[a-zA-Z]',  # Numbers with letters like "1.2a"
        r'^\([a-z0-9]+\)',              # Parenthetical labels like "(2S)"
        r'^\[[a-z0-9]+\]',              # Bracket labels
        
        # Very specific chemical names
        r'butyl',
        r'propyl',
        r'amino',
        r'hydroxy',
        r'glycol',
        r'ester',
        r'ether',
        r'oxide',
        r'polymer',
        r'peptide'
    ]
    
    # Check if heading matches any blacklist pattern
    for pattern in blacklist_patterns:
        if re.search(pattern, heading_lower):
            if debug:
                print(f"EXCLUDED (blacklist): {heading_text} - Pattern: {pattern}")
            return False
    
    # Length-based criteria
    if len(heading_text) > 25:
        if debug:
            print(f"EXCLUDED (too long): {heading_text}")
        return False
    
    # If nothing matched specifically, check if it has characteristics of a main heading
    if heading_text.istitle() and len(heading_text.split()) <= 4:
        if debug:
            print(f"MAIN HEADING (fallback): {heading_text}")
        return True
        
    # Default to exclusion for anything that doesn't match our positive criteria
    if debug:
        print(f"EXCLUDED (default): {heading_text}")
    return False

def extract_bibliographic_data(root):
    """Extract bibliographic data from patent XML."""
    bibliographic_data = {}
    
    # Extract root document attributes
    if root.tag == 'ep-patent-document':
        bibliographic_data.update({
            "doc_id": root.get('id'),
            "file": root.get('file'),
            "language": root.get('lang'),
            "country": root.get('country'),
            "doc_number": root.get('doc-number'),
            "kind_code": root.get('kind'),
            "correction_code": root.get('correction-code'),
            "publication_date": root.get('date-publ'),
            "status": root.get('status'),
            "dtd_version": root.get('dtd-version')
        })
    
    # Process SDOBI (Standard Document BIbliography) section
    sdobi = root.find('.//SDOBI')
    if sdobi:
        # Document type/title (B121)
        doc_type_elem = sdobi.find('.//B121')
        if doc_type_elem is not None and doc_type_elem.text:
            bibliographic_data["document_type"] = doc_type_elem.text
        
        # Document kind code (B130)
        kind_elem = sdobi.find('.//B130')
        if kind_elem is not None and kind_elem.text:
            bibliographic_data["kind"] = kind_elem.text

        # Publication date (B140)
        pub_date_elem = sdobi.find('.//B140/date')
        if pub_date_elem is not None and pub_date_elem.text:
            bibliographic_data["publication_date_full"] = pub_date_elem.text
        
        # Correction information (B150)
        if sdobi.find('.//B150') is not None:
            correction_info = {}
            correction_code = sdobi.find('.//B151')
            if correction_code is not None and correction_code.text:
                correction_info["correction_code"] = correction_code.text
            
            # Extract correction details from B154 and B155
            for section in ['B154', 'B155']:
                details = []
                section_elems = sdobi.findall(f'.//{section}/*')
                for i in range(0, len(section_elems), 2):
                    if i+1 < len(section_elems):
                        lang_elem = section_elems[i]
                        text_elem = section_elems[i+1]
                        if lang_elem.text and text_elem.text:
                            details.append({
                                "language": lang_elem.text,
                                "text": text_elem.text
                            })
                
                if details:
                    correction_info[f"correction_details_{section.lower()}"] = details
            
            if correction_info:
                bibliographic_data["correction_information"] = correction_info
        
        # Application information
        app_num = sdobi.find('.//B210')
        if app_num is not None and app_num.text:
            bibliographic_data["application_number"] = app_num.text
        
        app_date = sdobi.find('.//B220/date')
        if app_date is not None and app_date.text:
            bibliographic_data["application_date"] = app_date.text
        
        # Priority information (B300)
        priorities = []
        for i, priority_num in enumerate(sdobi.findall('.//B310')):
            if priority_num.text:
                priority = {"number": priority_num.text}
                
                # Find corresponding date and country
                if i < len(sdobi.findall('.//B320/date')):
                    date_elem = sdobi.findall('.//B320/date')[i]
                    if date_elem is not None and date_elem.text:
                        priority["date"] = date_elem.text
                
                if i < len(sdobi.findall('.//B330/ctry')):
                    country_elem = sdobi.findall('.//B330/ctry')[i]
                    if country_elem is not None and country_elem.text:
                        priority["country"] = country_elem.text
                
                priorities.append(priority)
        
        if priorities:
            bibliographic_data["priorities"] = priorities
        
        # Bulletin information
        for field in ['B405', 'B430', 'B450']:
            bulletin_elem = sdobi.find(f'.//{field}')
            if bulletin_elem is not None:
                date_elem = bulletin_elem.find('./date')
                bnum_elem = bulletin_elem.find('./bnum')
                
                info = {}
                if date_elem is not None and date_elem.text:
                    info["date"] = date_elem.text
                if bnum_elem is not None and bnum_elem.text:
                    info["bulletin_number"] = bnum_elem.text
                
                if info:
                    field_name = {
                        'B405': 'corrigendum_bulletin',
                        'B430': 'publication_bulletin',
                        'B450': 'grant_bulletin'
                    }.get(field)
                    bibliographic_data[field_name] = info
        
        # IPC (International Patent Classification) information
        ipc_classes = []
        for ipc_elem in sdobi.findall('.//classification-ipcr'):
            text_elem = ipc_elem.find('./text')
            if text_elem is not None and text_elem.text:
                ipc_classes.append(text_elem.text.strip())
        
        if ipc_classes:
            bibliographic_data["ipc_classes"] = ipc_classes
        
        # CPC (Cooperative Patent Classification) information
        cpc_classes = []
        for cpc_elem in sdobi.findall('.//classification-cpc'):
            text_elem = cpc_elem.find('./text')
            if text_elem is not None and text_elem.text:
                cpc_classes.append(text_elem.text.strip())
        
        if cpc_classes:
            bibliographic_data["cpc_classes"] = cpc_classes
        
        # Title information
        title_info = {}
        title_section = sdobi.find('.//B540')
        if title_section is not None:
            for i, lang_elem in enumerate(title_section.findall('.//B541')):
                if lang_elem.text and i < len(title_section.findall('.//B542')):
                    title_elem = title_section.findall('.//B542')[i]
                    if title_elem is not None and title_elem.text:
                        title_info[lang_elem.text] = title_elem.text
        
        if title_info:
            bibliographic_data["title"] = title_info
        
        # Extract inventors, applicants, representatives, etc. (code unchanged)
        # ... [keep the rest of your existing bibliographic extraction code] ...
        
        # Extract inventors information
        inventors = []
        for inv_elem in sdobi.findall('.//B721'):
            inventor = {}
            
            name_elem = inv_elem.find('./snm')
            if name_elem is not None and name_elem.text:
                inventor["name"] = name_elem.text
            
            # Extract address information
            addr_elem = inv_elem.find('./adr')
            if addr_elem is not None:
                address = {}
                
                street_elem = addr_elem.find('./str')
                if street_elem is not None and street_elem.text:
                    address["street"] = street_elem.text
                
                city_elem = addr_elem.find('./city')
                if city_elem is not None and city_elem.text:
                    address["city"] = city_elem.text
                
                country_elem = addr_elem.find('./ctry')
                if country_elem is not None and country_elem.text:
                    address["country"] = country_elem.text
                
                if address:
                    inventor["address"] = address
            
            if inventor:
                inventors.append(inventor)
        
        if inventors:
            bibliographic_data["inventors"] = inventors
        
        # Extract applicant information
        applicants = []
        for app_elem in sdobi.findall('.//B731'):
            applicant = {}
            
            name_elem = app_elem.find('./snm')
            if name_elem is not None and name_elem.text:
                applicant["name"] = name_elem.text
            
            id_elem = app_elem.find('./iid')
            if id_elem is not None and id_elem.text:
                applicant["id"] = id_elem.text
            
            ref_elem = app_elem.find('./irf')
            if ref_elem is not None and ref_elem.text:
                applicant["reference"] = ref_elem.text
            
            # Extract address information
            addr_elem = app_elem.find('./adr')
            if addr_elem is not None:
                address = {}
                
                street_elem = addr_elem.find('./str')
                if street_elem is not None and street_elem.text:
                    address["street"] = street_elem.text
                
                city_elem = addr_elem.find('./city')
                if city_elem is not None and city_elem.text:
                    address["city"] = city_elem.text
                
                country_elem = addr_elem.find('./ctry')
                if country_elem is not None and country_elem.text:
                    address["country"] = country_elem.text
                
                if address:
                    applicant["address"] = address
            
            if applicant:
                applicants.append(applicant)
        
        if applicants:
            bibliographic_data["applicants"] = applicants
        
        # Extract representative/agent information
        representatives = []
        for rep_elem in sdobi.findall('.//B741'):
            representative = {}
            
            name_elem = rep_elem.find('./snm')
            if name_elem is not None and name_elem.text:
                representative["name"] = name_elem.text
            
            id_elem = rep_elem.find('./iid')
            if id_elem is not None and id_elem.text:
                representative["id"] = id_elem.text
            
            # Extract address information
            addr_elem = rep_elem.find('./adr')
            if addr_elem is not None:
                address = {}
                
                street_elem = addr_elem.find('./str')
                if street_elem is not None and street_elem.text:
                    address["street"] = street_elem.text
                
                city_elem = addr_elem.find('./city')
                if city_elem is not None and city_elem.text:
                    address["city"] = city_elem.text
                
                country_elem = addr_elem.find('./ctry')
                if country_elem is not None and country_elem.text:
                    address["country"] = country_elem.text
                
                if address:
                    representative["address"] = address
            
            if representative:
                representatives.append(representative)
        
        if representatives:
            bibliographic_data["representatives"] = representatives
        
        # Extract designated states information
        designated_states = []
        states_elem = sdobi.find('.//B840')
        if states_elem is not None:
            for state_elem in states_elem.findall('./ctry'):
                if state_elem.text:
                    designated_states.append(state_elem.text)
        
        if designated_states:
            bibliographic_data["designated_states"] = designated_states
    
    # Process abstract
    abstract = root.find('.//abstract')
    if abstract:
        abstract_text = ""
        for p in abstract.findall('.//p'):
            p_text = p.text or ""
            # Remove XML comments
            p_text = re.sub(r'<!--.*?-->', '', p_text)
            if p_text.strip():
                abstract_text += p_text.strip() + " "
        
        if abstract_text.strip():
            bibliographic_data["abstract"] = abstract_text.strip()
    
    return bibliographic_data

def extract_main_sections(root, debug=False):
    """
    Extract main sections from patent XML, with stricter filtering to exclude 
    ALL compounds and examples.
    """
    main_sections = []
    
    # Find description section
    description = root.find('.//description')
    if description:
        # First, collect all elements in order
        ordered_elements = []
        for elem in description:
            if elem.tag in ['heading', 'p']:
                ordered_elements.append(elem)
        
        # Process elements and associate paragraphs with headings
        current_main_section = None
        
        for elem in ordered_elements:
            if elem.tag == 'heading':
                heading_id = elem.get('id')
                heading_text = extract_heading_text(elem)
                
                # Use the enhanced is_main_heading function with debug option
                if heading_text and is_main_heading(heading_text, debug=debug):
                    current_main_section = {
                        "heading_id": heading_id,
                        "heading_text": heading_text,
                        "paragraphs": []
                    }
                    main_sections.append(current_main_section)
            
            elif elem.tag == 'p' and current_main_section:
                p_id = elem.get('id')
                p_num = elem.get('num')
                p_text = elem.text or ""
                
                # Remove XML comments
                p_text = re.sub(r'<!--.*?-->', '', p_text)
                
                if p_text.strip():
                    current_main_section["paragraphs"].append({
                        "p_id": p_id,
                        "p_number": p_num,
                        "text": p_text.strip()
                    })
    
    return main_sections

def extract_claims(root):
    """Extract claims from patent XML."""
    claims = []
    
    # Find claims section
    claims_section = root.find('.//claims')
    if claims_section:
        for claim_elem in claims_section.findall('.//claim'):
            claim_num = claim_elem.get('num')
            claim_text = ""
            
            # Extract claim text from various possible structures
            for text_elem in claim_elem.iter():
                if text_elem.tag in ['claim-text', 'p'] and text_elem.text:
                    text = text_elem.text.strip()
                    if text:
                        claim_text += text + " "
            
            claim_text = claim_text.strip()
            if claim_text:
                claims.append({
                    "claim_number": claim_num,
                    "text": claim_text
                })
    
    return claims

def process_patent_xml(xml_file_path, output_file=None, debug=False):
    """
    Process patent XML and extract structured data with enhanced filtering.
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Initialize result structure
        patent_data = {
            "bibliographic_data": {},
            "main_sections": [],
            "claims": []
        }
        
        # Extract data using the specialized functions
        patent_data["bibliographic_data"] = extract_bibliographic_data(root)
        patent_data["main_sections"] = extract_main_sections(root, debug=debug)
        patent_data["claims"] = extract_claims(root)
        
        # Save to JSON file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(patent_data, f, indent=2, ensure_ascii=False)
            print(f"Patent data saved to {output_file}")
        
        return patent_data
    
    except Exception as e:
        print(f"Error processing patent XML: {e}")
        import traceback
        traceback.print_exc()
        return {
            "bibliographic_data": {},
            "main_sections": [],
            "claims": []
        }

def process_xml_files_list(xml_file_paths, debug=False):
    """
    Process a list of XML file paths and save JSON outputs to corresponding processed directory structure.
    
    Args:
        xml_file_paths: List of XML file paths from get_all_epo_file_paths() or similar
        debug: Whether to enable debug output
    
    Example:
        Input:  ../data/raw/EPO/EPRTBJV2025000023001001/EPW1B9/EP18823397W1B9/EP18823397W1B9.xml
        Output: ../data/processed/EPO/EPRTBJV2025000023001001/EPW1B9/EP18823397W1B9/EP18823397W1B9.json
    """
    print(f"🚀 Processing {len(xml_file_paths)} XML files...")
    
    processed_count = 0
    error_count = 0
    
    for i, xml_path in enumerate(xml_file_paths, 1):
        try:
            # Convert raw path to processed path
            processed_path = convert_raw_to_processed_path(xml_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(processed_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Progress indicator
            if i % 100 == 0 or i == 1:
                print(f"📄 Processing {i}/{len(xml_file_paths)}: {os.path.basename(xml_path)}")
            
            # Process the XML file
            result = process_patent_xml(xml_path, processed_path, debug=debug)
            
            if result and (result.get('bibliographic_data') or result.get('main_sections') or result.get('claims')):
                processed_count += 1
            else:
                error_count += 1
                print(f"⚠️ Warning: No data extracted from {os.path.basename(xml_path)}")
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {os.path.basename(xml_path)}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Results:")
    print(f"  • Successfully processed: {processed_count}")
    print(f"  • Errors: {error_count}")
    print(f"  • Total files: {len(xml_file_paths)}")


def convert_raw_to_processed_path(raw_xml_path):
    """
    Convert a raw XML file path to the corresponding processed JSON file path.
    
    Args:
        raw_xml_path: Path like ../data/raw/EPO/EPRTBJV.../folder/file.xml
        
    Returns:
        Processed path like ../data/processed/EPO/EPRTBJV.../folder/file.json
    """
    # Convert to Path object for easier manipulation
    from pathlib import Path
    
    raw_path = Path(raw_xml_path)
    
    # Replace 'raw' with 'processed' in the path
    path_parts = list(raw_path.parts)
    
    # Find and replace 'raw' with 'processed'
    for i, part in enumerate(path_parts):
        if part == 'raw':
            path_parts[i] = 'processed'
            break
    
    # Change extension from .xml to .json
    filename = raw_path.stem + '.json'
    path_parts[-1] = filename
    
    # Reconstruct the path
    processed_path = Path(*path_parts)
    
    return str(processed_path)

# Updated process_directory function (optional - for backward compatibility)
def process_directory(directory_path, output_dir=None, debug=False):
    """
    Process all XML files in a directory with enhanced filtering.
    
    Args:
        directory_path: Directory containing XML files
        output_dir: Output directory (if None, uses same as input)
        debug: Whether to enable debug output
    """
    if output_dir is None:
        output_dir = directory_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files in the directory
    xml_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]
    xml_file_paths = [os.path.join(directory_path, f) for f in xml_files]
    
    print(f"Found {len(xml_files)} XML files in {directory_path}")
    
    # Use the new function to process the list
    process_xml_files_list(xml_file_paths, debug=debug)