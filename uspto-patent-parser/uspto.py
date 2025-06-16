from uspto import *
from parsers.parse_txt_file import *
from parsers.parse_xml_v2_file import *
from parsers.parse_xml_v4_file import *
from parsers.parse_xml_new_file import *
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
import requests, zipfile, io
import re
import os



def read_and_parse_yearly_data(year, data_items):
    if type(year) == int and year >= 1976 and year <= 2020: 
        list_of_files = requests.get(f'https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/')
        page_html_text = list_of_files.text
        html_parser = BeautifulSoup(page_html_text, 'html.parser')
        href_files_list = []
        for tr in html_parser.find_all('table')[1].find_all('a'):
            if tr['href'] != None: 
                href_files_list.append(tr['href'])
        if year == 2001:
            href_files_list = list(filter(lambda x: x.endswith('.zip') and 'aps' in x, href_files_list))
        else:
            href_files_list = list(filter(lambda x: x.endswith('.zip'), href_files_list))
        full_yearly_data = []
        print(f'{len(href_files_list)} files found')
        current_file_number = 0
        for file in href_files_list:
            current_file_number += 1
            url = f'https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/{file}'
            print(f'Parsing file {current_file_number}/{len(href_files_list)} - {url}')
            if year <= 2001:
                patent_data = read_data_from_url_txt(url)
                for patent in patent_data:
                    full_yearly_data.append(parse_txt_patent_data(patent,source_url = url,data_items_list=data_items))
            elif year in [2002,2003,2004]:
                patent_data = read_data_from_url_xml_2(url)
                for patent in patent_data:
                    root_tree = ElementTree(fromstring(patent))
                    full_yearly_data.append(parse_patent_data_xml_2(root_tree,source_url = url,data_items_list=data_items))
            elif year > 2004:
                patent_data = read_data_from_url_xml_4(url)
                for patent in patent_data:
                    root_tree = ElementTree(fromstring(patent))
                    full_yearly_data.append(parse_patent_data_xml_4(root_tree,source_url = url,data_items_list=data_items))
            time.sleep(30 + random.choice(range(10)))
        return full_yearly_data
    else:
        print(f'ERROR: Invalid year argument "{year}". Year must be an integer number between 1975 and 2020')
        
def download_file_to_disk(url,target_path):
    request_data = requests.get(url)
    zipped_files = zipfile.ZipFile(io.BytesIO(request_data.content))
    zipped_files.extractall(target_path)
    
def get_patent_files_by_year(year):
        list_of_files = requests.get(f'https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/')
        page_html_text = list_of_files.text
        html_parser = BeautifulSoup(page_html_text, 'html.parser')
        href_files_list = []
        for tr in html_parser.find_all('table')[1].find_all('a'):
            if tr['href'] != None: 
                href_files_list.append(tr['href'])
        if year == 2001:
            href_files_list = list(filter(lambda x: x.endswith('.zip') and 'aps' in x, href_files_list))
        else:
            href_files_list = list(filter(lambda x: x.endswith('.zip'), href_files_list))
        return href_files_list
    
def read_and_parse_from_url(url,data_items):
        m = re.search('(?<=fulltext/)\d+', url)
        year = int(m.group(0))
        full_yearly_data = []
        if year <= 2001:
            raw_patent_data = read_data_from_url_txt(url)
            for patent in raw_patent_data:
                if 'URL' in data_items:
                    full_yearly_data.append(parse_txt_patent_data(patent,source_url = url,data_items_list=data_items))
                else:
                    full_yearly_data.append(parse_txt_patent_data(patent,data_items_list=data_items))
        elif year in [2002,2003,2004]:
            raw_patent_data = read_data_from_url_xml_2(url)
            for patent in raw_patent_data:
                root_tree = ElementTree(fromstring(patent))
                if 'URL' in data_items:
                    full_yearly_data.append(parse_patent_data_xml_2(root_tree,source_url = url,data_items_list=data_items))
                else:
                    full_yearly_data.append(parse_patent_data_xml_2(root_tree,data_items_list=data_items))
        elif year > 2004:
            raw_patent_data = read_data_from_url_xml_4(url)
            for patent in raw_patent_data:
                root_tree = ElementTree(fromstring(patent))
                if 'URL' in data_items:
                    full_yearly_data.append(parse_patent_data_xml_4(root_tree,source_url = url,data_items_list=data_items))
                else:
                    full_yearly_data.append(parse_patent_data_xml_4(root_tree,data_items_list=data_items))
        return full_yearly_data
    
def read_and_parse_file_from_disk(path_to_file,data_items,extension):
    if extension == 'txt':
        data = read_and_parse_txt_from_disk(path_to_file,data_items)
        return data
    elif extension == 'xml2':
        data = read_and_parse_xml2_from_disk(path_to_file,data_items)
        return data
    elif extension == 'xml4':
        data = read_and_parse_xml4_from_disk(path_to_file,data_items)
        return data
    
def download_yearly_data(year,data_items):
    if type(year) == int and year >= 1976: 
        list_of_files = requests.get(f'https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/')
        page_html_text = list_of_files.text
        html_parser = BeautifulSoup(page_html_text, 'html.parser')
        href_files_list = []
        for tr in html_parser.find_all('table')[1].find_all('a'):
            if tr['href'] != None: 
                href_files_list.append(tr['href'])
        if year == 2001:
            href_files_list = list(filter(lambda x: x.endswith('.zip') and 'aps' in x, href_files_list))
        else:
            href_files_list = list(filter(lambda x: x.endswith('.zip'), href_files_list))
        full_yearly_data = []
        print(f'{len(href_files_list)} files found')
        current_file_number = 0
        for file in href_files_list:
            current_file_number += 1
            url = f'https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/{file}'
            print(f'Parsing file {current_file_number}/{len(href_files_list)} - {url}')
            if year <= 2001:
                patent_data = read_data_from_url_txt(url)
                for patent in patent_data:
                    full_yearly_data.append(parse_txt_patent_data(patent,source_url = url,data_items_list=data_items))
            elif year in [2002,2003,2004]:
                patent_data = read_data_from_url_xml_2(url)
                for patent in patent_data:
                    root_tree = ElementTree(fromstring(patent))
                    full_yearly_data.append(parse_patent_data_xml_2(root_tree,source_url = url,data_items_list=data_items))
            elif year > 2004:
                patent_data = read_data_from_url_xml_4(url)
                for patent in patent_data:
                    root_tree = ElementTree(fromstring(patent))
                    full_yearly_data.append(parse_patent_data_xml_4(root_tree,source_url = url,data_items_list=data_items))
            time.sleep(30 + random.choice(range(10)))
        return full_yearly_data
    else:
        print(f'ERROR: Invalid year argument "{year}". Year must be an integer number greater than or equal 1976')

def parse_local_zip_file(zip_path, data_items, extract_dir="extracted_files"):
    os.makedirs(extract_dir, exist_ok=True)
    all_patent_data = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            # Extract file to disk
            extracted_path = os.path.join(extract_dir, name)
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
            with open(extracted_path, "wb") as f:
                f.write(zf.read(name))
            # Decide parser based on file extension
            # print(f"Processing file: {name}")
            data = parse_uspto_xml_file("/Users/level_3/hienptx/uspto-patent-data-parser/extracted_files/US20250185527A1-20250612/US20250185527A1-20250612.xml")
            # if name.endswith(".xml"):
            #     data = read_and_parse_file_from_disk(extracted_path, data_items, extension="xml4")
            # elif name.endswith(".txt"):
            #     data = read_and_parse_file_from_disk(extracted_path, data_items, extension="txt")
            # else:
            #     # print(f"Unknown file type: {name}")
            #     continue
            all_patent_data.extend(data)
    return all_patent_data

# Example usage:
if __name__ == '__main__':
    zip_path = "US20250185527A1-20250612.ZIP"  # Replace with your zip file name

    # List all possible fields your parser can extract
    data_items = [
        'PATENT_NUM', 'TITLE', 'ABSTRACT', 'INVENTORS', 'ASSIGNEE',
        'FILING_DATE', 'GRANT_DATE', 'APPLICATION_NUM', 'CPC', 'IPC',
        'US_CLASS', 'CLAIMS', 'DESCRIPTION', 'URL'
        # Add/remove fields as supported by your parser
    ]
    results = parse_local_zip_file(zip_path, data_items)
    print(f"Parsed {len(results)} patents.")
    if results:
        print("First patent:", results[0])
        print(json.dumps(results[0], indent=2, ensure_ascii=False))