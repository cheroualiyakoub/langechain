import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from xml.etree.ElementTree import fromstring, ElementTree
import re
import time
import json
import requests
from io import BytesIO
from zipfile import ZipFile

# Updated paths for patent applications (2025 format)
# Root element for patent applications
patent_application_root = './us-patent-application'
patent_grant_root = './us-patent-grant'

# Bibliographic data paths - updated for both applications and grants
app_bibliographic_base = './us-bibliographic-data-application'
grant_bibliographic_base = './us-bibliographic-data-grant'

# Publication reference paths
app_publication_ref = f'{app_bibliographic_base}/publication-reference/document-id'
grant_publication_ref = f'{grant_bibliographic_base}/publication-reference/document-id'

# Application reference paths
app_application_ref = f'{app_bibliographic_base}/application-reference/document-id'
grant_application_ref = f'{grant_bibliographic_base}/application-reference/document-id'

# Updated classification paths (current format)
app_classifications_cpc = f'{app_bibliographic_base}/classifications-cpc/classification-cpc'
grant_classifications_cpc = f'{grant_bibliographic_base}/classifications-cpc/classification-cpc'

app_classifications_ipcr = f'{app_bibliographic_base}/classifications-ipcr/classification-ipcr'
grant_classifications_ipcr = f'{grant_bibliographic_base}/classifications-ipcr/classification-ipcr'

app_classification_national = f'{app_bibliographic_base}/classification-national'
grant_classification_national = f'{grant_bibliographic_base}/classification-national'

# Field of search paths
app_field_of_search = f'{app_bibliographic_base}/field-of-search'
grant_field_of_search = f'{grant_bibliographic_base}/field-of-search'

# Party information paths (updated structure)
app_parties_base = f'{app_bibliographic_base}/us-parties'
grant_parties_base = f'{grant_bibliographic_base}/us-parties'

# Alternative parties paths for older/different structures
app_parties_alt = f'{app_bibliographic_base}/parties'
grant_parties_alt = f'{grant_bibliographic_base}/parties'

# Inventor paths
app_inventors = f'{app_parties_base}/inventors/inventor'
grant_inventors = f'{grant_parties_base}/inventors/inventor'
app_inventors_alt = f'{app_parties_alt}/inventors/inventor'
grant_inventors_alt = f'{grant_parties_alt}/inventors/inventor'

# Applicant paths (for applications)
app_applicants = f'{app_parties_base}/us-applicants/us-applicant'
app_applicants_alt = f'{app_parties_alt}/applicants/applicant'

# Assignee paths
app_assignees = f'{app_bibliographic_base}/assignees/assignee'
grant_assignees = f'{grant_bibliographic_base}/assignees/assignee'

# Agent/Attorney paths
app_agents = f'{app_parties_base}/agents/agent'
grant_agents = f'{grant_parties_base}/agents/agent'
app_agents_alt = f'{app_parties_alt}/agents/agent'
grant_agents_alt = f'{grant_parties_alt}/agents/agent'

# Citation paths (updated)
app_citations = f'{app_bibliographic_base}/us-references-cited/us-citation'
grant_citations = f'{grant_bibliographic_base}/us-references-cited/us-citation'
app_citations_alt = f'{app_bibliographic_base}/references-cited/citation'
grant_citations_alt = f'{grant_bibliographic_base}/references-cited/citation'

# Priority claims
app_priority_claims = f'{app_bibliographic_base}/priority-claims/priority-claim'
grant_priority_claims = f'{grant_bibliographic_base}/priority-claims/priority-claim'

# Title paths
app_invention_title = f'{app_bibliographic_base}/invention-title'
grant_invention_title = f'{grant_bibliographic_base}/invention-title'

# Abstract and description paths
abstract_path = './abstract'
description_path = './description'
claims_path = './claims/claim'
drawings_path = './drawings'

def detect_document_type(root_tree):
    """Detect if the document is a patent application or grant"""
    root_tag = root_tree.getroot().tag if hasattr(root_tree, 'getroot') else root_tree.tag
    if root_tag == 'us-patent-application':
        return 'application'
    elif root_tag == 'us-patent-grant':
        return 'grant'
    else:
        return 'unknown'

def get_paths_for_document_type(doc_type):
    """Return appropriate paths based on document type"""
    if doc_type == 'application':
        return {
            'bibliographic_base': app_bibliographic_base,
            'publication_ref': app_publication_ref,
            'application_ref': app_application_ref,
            'classifications_cpc': app_classifications_cpc,
            'classifications_ipcr': app_classifications_ipcr,
            'classification_national': app_classification_national,
            'field_of_search': app_field_of_search,
            'parties_base': app_parties_base,
            'parties_alt': app_parties_alt,
            'inventors': app_inventors,
            'inventors_alt': app_inventors_alt,
            'applicants': app_applicants,
            'applicants_alt': app_applicants_alt,
            'assignees': app_assignees,
            'agents': app_agents,
            'agents_alt': app_agents_alt,
            'citations': app_citations,
            'citations_alt': app_citations_alt,
            'priority_claims': app_priority_claims,
            'invention_title': app_invention_title
        }
    else:  # grant
        return {
            'bibliographic_base': grant_bibliographic_base,
            'publication_ref': grant_publication_ref,
            'application_ref': grant_application_ref,
            'classifications_cpc': grant_classifications_cpc,
            'classifications_ipcr': grant_classifications_ipcr,
            'classification_national': grant_classification_national,
            'field_of_search': grant_field_of_search,
            'parties_base': grant_parties_base,
            'parties_alt': grant_parties_alt,
            'inventors': grant_inventors,
            'inventors_alt': grant_inventors_alt,
            'applicants': None,  # Grants don't have applicants in the same way
            'applicants_alt': None,
            'assignees': grant_assignees,
            'agents': grant_agents,
            'agents_alt': grant_agents_alt,
            'citations': grant_citations,
            'citations_alt': grant_citations_alt,
            'priority_claims': grant_priority_claims,
            'invention_title': grant_invention_title
        }

def safe_find_element(root_tree, path):
    """Safely find an element, returning None if not found"""
    try:
        return root_tree.find(path)
    except:
        return None

def safe_findall_elements(root_tree, path):
    """Safely find all elements, returning empty list if not found"""
    try:
        return root_tree.findall(path)
    except:
        return []

def get_text_content(element):
    """Safely get text content from an element"""
    if element is not None:
        if element.text:
            return element.text.strip()
        else:
            # Handle elements with mixed content
            return ''.join(element.itertext()).strip()
    return None

def read_and_parse_xml_from_disk(path_to_file, data_items):
    """Read and parse XML file from disk"""
    with open(path_to_file, 'r', encoding='utf-8', errors='ignore') as f:
        xml_content = f.read()
    
    # Split on XML declarations to separate individual patents
    xml_parts = xml_content.split('<?xml version="1.0" encoding="UTF-8"?>')
    xml_parts = [part.strip() for part in xml_parts if part.strip()]
    
    parsed_data = []
    for xml_part in xml_parts:
        try:
            root_tree = ElementTree(fromstring(xml_part))
            parsed_data.append(parse_patent_data_xml(root_tree, data_items_list=data_items))
        except Exception as e:
            print(f"Error parsing XML part: {e}")
            continue
    
    return parsed_data

def read_data_from_url_xml(url):
    """Download and extract XML data from USPTO ZIP file"""
    response = requests.get(url)
    read_url = ZipFile(BytesIO(response.content))
    file_name = list(filter(lambda file: '.xml' in file.lower(), read_url.namelist()))[0]
    
    data_bytes = read_url.open(file_name).read()
    data_string = data_bytes.decode('utf-8', errors='ignore')
    
    # Split XML documents
    patent_documents = data_string.split('<?xml version="1.0" encoding="UTF-8"?>')
    patent_documents = [doc.strip() for doc in patent_documents if doc.strip()]
    
    read_url.close()
    return patent_documents

def get_patent_identification_data(root_tree, paths):
    """Extract basic patent identification information"""
    document_data = {}
    
    # Publication information
    publication_info = safe_find_element(root_tree, paths['publication_ref'])
    if publication_info is not None:
        for element in publication_info:
            if element.text:
                document_data[element.tag] = element.text.strip()
    
    # Application information
    application_info = safe_find_element(root_tree, paths['application_ref'])
    if application_info is not None:
        for element in application_info:
            if element.text:
                document_data[f"application_{element.tag}"] = element.text.strip()
        
        # Get application type attribute if present
        if application_info.attrib and 'appl-type' in application_info.attrib:
            document_data['application_type'] = application_info.attrib['appl-type']
    
    # Invention title
    title_element = safe_find_element(root_tree, paths['invention_title'])
    if title_element is not None:
        document_data['invention_title'] = get_text_content(title_element)
    
    return document_data

def get_classification_data(root_tree, paths):
    """Extract classification information"""
    classification_data = {}
    
    # CPC Classifications
    cpc_classifications = safe_findall_elements(root_tree, paths['classifications_cpc'])
    if cpc_classifications:
        cpc_list = []
        for cpc in cpc_classifications:
            cpc_data = {}
            for child in cpc:
                if child.text:
                    cpc_data[child.tag] = child.text.strip().replace(" ", "")
            if cpc_data:
                cpc_list.append(cpc_data)
        if cpc_list:
            classification_data['cpc_classifications'] = cpc_list
    
    # IPCR Classifications
    ipcr_classifications = safe_findall_elements(root_tree, paths['classifications_ipcr'])
    if ipcr_classifications:
        ipcr_list = []
        for ipcr in ipcr_classifications:
            ipcr_data = {}
            # Common IPCR fields
            ipcr_fields = ['ipc-version-indicator', 'classification-level', 'section', 
                          'class', 'subclass', 'main-group', 'subgroup', 
                          'symbol-position', 'classification-value', 'action-date', 
                          'generating-office', 'classification-status', 'classification-data-source']
            
            for field in ipcr_fields:
                element = safe_find_element(ipcr, field)
                if element is not None:
                    ipcr_data[field] = get_text_content(element)
                
                # Also check for nested date/country elements
                if field == 'action-date':
                    date_elem = safe_find_element(ipcr, f'{field}/date')
                    if date_elem is not None:
                        ipcr_data[f'{field}_date'] = get_text_content(date_elem)
                elif field == 'generating-office':
                    country_elem = safe_find_element(ipcr, f'{field}/country')
                    if country_elem is not None:
                        ipcr_data[f'{field}_country'] = get_text_content(country_elem)
            
            if ipcr_data:
                ipcr_list.append(ipcr_data)
        
        if ipcr_list:
            classification_data['ipcr_classifications'] = ipcr_list
    
    # National Classifications
    national_class = safe_find_element(root_tree, paths['classification_national'])
    if national_class is not None:
        national_data = {}
        for element in national_class:
            if element.text:
                national_data[element.tag] = element.text.strip().replace(" ", "")
        if national_data:
            classification_data['national_classification'] = national_data
    
    return classification_data

def get_party_data(root_tree, paths, party_type):
    """Extract party information (inventors, applicants, agents, assignees)"""
    parties_list = []
    
    if party_type == 'inventors':
        party_elements = safe_findall_elements(root_tree, paths['inventors'])
        if not party_elements:
            party_elements = safe_findall_elements(root_tree, paths['inventors_alt'])
    elif party_type == 'applicants':
        if paths['applicants']:
            party_elements = safe_findall_elements(root_tree, paths['applicants'])
            if not party_elements and paths['applicants_alt']:
                party_elements = safe_findall_elements(root_tree, paths['applicants_alt'])
        else:
            party_elements = []
    elif party_type == 'agents':
        party_elements = safe_findall_elements(root_tree, paths['agents'])
        if not party_elements:
            party_elements = safe_findall_elements(root_tree, paths['agents_alt'])
    elif party_type == 'assignees':
        party_elements = safe_findall_elements(root_tree, paths['assignees'])
    else:
        party_elements = []
    
    for party in party_elements:
        party_data = {}
        
        # Get attributes
        if party.attrib:
            party_data.update(party.attrib)
        
        # Address information
        address_elem = safe_find_element(party, 'addressbook/address')
        if address_elem is not None:
            for addr_item in address_elem:
                if addr_item.text:
                    party_data[addr_item.tag] = addr_item.text.strip()
        
        # Name information
        name_fields = ['addressbook/orgname', 'addressbook/first-name', 'addressbook/last-name',
                      'addressbook/role', 'orgname', 'first-name', 'last-name', 'role']
        
        for field in name_fields:
            elem = safe_find_element(party, field)
            if elem is not None:
                party_data[field.split('/')[-1]] = get_text_content(elem)
        
        # Residence information
        residence_elem = safe_find_element(party, 'residence/country')
        if residence_elem is not None:
            party_data['residence_country'] = get_text_content(residence_elem)
        
        if party_data:
            parties_list.append(party_data)
    
    return parties_list

def get_citation_data(root_tree, paths):
    """Extract citation information"""
    citations = safe_findall_elements(root_tree, paths['citations'])
    if not citations:
        citations = safe_findall_elements(root_tree, paths['citations_alt'])
    
    patent_citations = []
    non_patent_citations = []
    
    for citation in citations:
        citation_data = {}
        
        # Patent citations
        patcit_elem = safe_find_element(citation, 'patcit')
        if patcit_elem is not None:
            doc_id_elem = safe_find_element(patcit_elem, 'document-id')
            if doc_id_elem is not None:
                for element in doc_id_elem:
                    if element.text:
                        citation_data[element.tag] = element.text.strip()
            
            # Category
            category_elem = safe_find_element(citation, 'category')
            if category_elem is not None:
                citation_data['category'] = get_text_content(category_elem)
            
            if citation_data:
                patent_citations.append(citation_data)
        
        # Non-patent citations
        nplcit_elem = safe_find_element(citation, 'nplcit')
        if nplcit_elem is not None:
            othercit_elem = safe_find_element(nplcit_elem, 'othercit')
            if othercit_elem is not None:
                npl_data = {'npl_citation': get_text_content(othercit_elem)}
                
                category_elem = safe_find_element(citation, 'category')
                if category_elem is not None:
                    npl_data['category'] = get_text_content(category_elem)
                
                non_patent_citations.append(npl_data)
    
    return patent_citations, non_patent_citations

def get_priority_data(root_tree, paths):
    """Extract priority claim information"""
    priority_claims = safe_findall_elements(root_tree, paths['priority_claims'])
    priority_data = []
    
    for claim in priority_claims:
        claim_data = {}
        
        # Get attributes
        if claim.attrib:
            claim_data.update(claim.attrib)
        
        # Get child elements
        fields = ['country', 'doc-number', 'date']
        for field in fields:
            elem = safe_find_element(claim, field)
            if elem is not None:
                claim_data[field] = get_text_content(elem)
        
        if claim_data:
            priority_data.append(claim_data)
    
    return priority_data

def get_abstract_data(root_tree):
    """Extract abstract information"""
    abstract_elements = safe_findall_elements(root_tree, f'{abstract_path}/p')
    abstract_data = []
    
    for paragraph in abstract_elements:
        text = get_text_content(paragraph)
        if text:
            abstract_data.append(text)
    
    return abstract_data

def get_description_data(root_tree):
    """Extract detailed description with full paragraph text, including nested tags."""
    description_elem = safe_find_element(root_tree, description_path)
    description_data = {'general_description_paragraphs': []}

    def get_full_text(element):
        """Return all text inside an element, including nested tags."""
        return ''.join(element.itertext()).strip() if element is not None else ''

    if description_elem is not None:
        current_heading = ''
        for item in description_elem:
            if item.tag == 'heading':
                current_heading = get_full_text(item)
                if current_heading:
                    description_data[current_heading] = []
            elif item.tag == 'p':
                text = get_full_text(item)
                if text:
                    if current_heading:
                        if current_heading not in description_data:
                            description_data[current_heading] = []
                        description_data[current_heading].append(text)
                    else:
                        description_data['general_description_paragraphs'].append(text)

    return description_data

def get_claims_data(root_tree):
    """Extract patent claims"""
    claims = safe_findall_elements(root_tree, claims_path)
    claims_list = []
    
    for claim in claims:
        claim_data = {}
        
        # Get attributes
        if claim.attrib:
            claim_data.update(claim.attrib)
        
        # Get claim text
        claim_texts = safe_findall_elements(claim, 'claim-text')
        if claim_texts:
            text_list = []
            for claim_text in claim_texts:
                text = get_text_content(claim_text)
                if text:
                    text_list.append(text)
            if text_list:
                claim_data['claim_text'] = text_list
        
        if claim_data:
            claims_list.append(claim_data)
    
    return claims_list

def parse_patent_data_xml(patent_tree_root, source_url=None, 
                         data_items_list=['INVT','ASSG','PRIP','CLAS','LREP','ABST','DETD','CLMS','CITA','OREF','URL']):
    """Main parsing function for patent XML data"""
    
    # Detect document type
    doc_type = detect_document_type(patent_tree_root)
    if doc_type == 'unknown':
        return {'error': 'Unknown document type'}
    
    # Get appropriate paths
    paths = get_paths_for_document_type(doc_type)
    
    filtered_data = {}
    filtered_data['document_type'] = doc_type
    
    # Basic bibliographic information (always included)
    filtered_data['bibliographic_information'] = get_patent_identification_data(patent_tree_root, paths)
    
    # Optional data based on requested items
    if 'URL' in data_items_list and source_url is not None:
        filtered_data['source_file'] = source_url
    
    if 'CLAS' in data_items_list:
        classifications = get_classification_data(patent_tree_root, paths)
        if classifications:
            filtered_data['classifications'] = classifications
    
    if 'INVT' in data_items_list:
        inventors = get_party_data(patent_tree_root, paths, 'inventors')
        if inventors:
            filtered_data['inventors'] = inventors
    
    if 'ASSG' in data_items_list:
        assignees = get_party_data(patent_tree_root, paths, 'assignees')
        if assignees:
            filtered_data['assignees'] = assignees
        
        # For applications, also get applicants
        if doc_type == 'application':
            applicants = get_party_data(patent_tree_root, paths, 'applicants')
            if applicants:
                filtered_data['applicants'] = applicants
    
    if 'LREP' in data_items_list:
        agents = get_party_data(patent_tree_root, paths, 'agents')
        if agents:
            filtered_data['legal_representatives'] = agents
    
    if 'CITA' in data_items_list or 'OREF' in data_items_list:
        patent_citations, non_patent_citations = get_citation_data(patent_tree_root, paths)
        if 'CITA' in data_items_list and patent_citations:
            filtered_data['patent_citations'] = patent_citations
        if 'OREF' in data_items_list and non_patent_citations:
            filtered_data['non_patent_citations'] = non_patent_citations
    
    if 'PRIP' in data_items_list:
        priority_data = get_priority_data(patent_tree_root, paths)
        if priority_data:
            filtered_data['priority_claims'] = priority_data
    
    if 'ABST' in data_items_list:
        abstract = get_abstract_data(patent_tree_root)
        if abstract:
            filtered_data['abstract'] = abstract
    
    if 'DETD' in data_items_list:
        description = get_description_data(patent_tree_root)
        if description and (description.get('general_description_paragraphs') or len(description) > 1):
            filtered_data['detailed_description'] = description
    
    if 'CLMS' in data_items_list:
        claims = get_claims_data(patent_tree_root)
        if claims:
            filtered_data['claims'] = claims
    
    return filtered_data

# Example usage function
def parse_uspto_xml_file(file_path, data_items=['INVT','ASSG','PRIP','CLAS','LREP','ABST','DETD','CLMS','CITA','OREF']):
    """
    Parse a USPTO XML file and return extracted data
    
    Args:
        file_path: Path to the XML file
        data_items: List of data items to extract
    
    Returns:
        List of parsed patent data dictionaries
    """
    return read_and_parse_xml_from_disk(file_path, data_items)

def parse_uspto_xml_from_url(url, data_items=['INVT','ASSG','PRIP','CLAS','LREP','ABST','DETD','CLMS','CITA','OREF']):
    """
    Download and parse USPTO XML from URL
    
    Args:
        url: URL to the USPTO ZIP file
        data_items: List of data items to extract
    
    Returns:
        List of parsed patent data dictionaries
    """
    xml_documents = read_data_from_url_xml(url)
    parsed_data = []
    
    for xml_doc in xml_documents:
        try:
            root_tree = ElementTree(fromstring(xml_doc))
            parsed_data.append(parse_patent_data_xml(root_tree, source_url=url, data_items_list=data_items))
        except Exception as e:
            print(f"Error parsing document: {e}")
            continue
    
    return parsed_data