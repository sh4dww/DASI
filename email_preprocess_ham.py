import numpy as np
import pandas as pd

import os, gc, re, csv, json

def calculate_seq_mean(array):
    total_sum = sum(array)
    mean = total_sum / len(array)
    return mean

folder = "ham"



#file = open("0041.2003-12-22.GP.spam.txt", "r", encoding = "utf-8")

def get_email_data(file):
    chars = [";", "(", "[", "!", "$", "#"]
    word_count = 0
    char_count = 0

    counts = {
        "make": 0,
        "address": 0,
        "all": 0,
        "3d": 0,
        "our": 0,
        "over": 0,
        "remove": 0,
        "internet": 0,
        "order": 0,
        "mail": 0,
        "receive": 0,
        "will": 0,
        "people": 0,
        "report": 0,
        "addresses": 0,
        "free": 0,
        "business": 0,
        "email": 0,
        "you": 0,
        "credit": 0,
        "your": 0,
        "font": 0,
        "000": 0,
        "money": 0,
        "hp": 0,
        "hpl": 0,
        "george": 0,
        "650": 0,
        "lab": 0,
        "labs": 0,
        "telnet": 0,
        "857": 0,
        "data": 0,
        "415": 0,
        "85": 0,
        "technology": 0,
        "1999": 0,
        "parts": 0,
        "pm": 0,
        "direct": 0,
        "cs": 0,
        "meeting": 0,
        "original": 0,
        "project": 0,
        "red": 0,
        "edu": 0,
        "table": 0,
        "conference": 0,
        ";": 0,
        "(": 0,
        "[": 0,
        "!": 0,
        "$": 0,
        "#": 0
    }

    email_data = {
        "word_freq_make": 0,
        "word_freq_address": 0,
        "word_freq_all": 0,
        "word_freq_3d": 0,
        "word_freq_our": 0,
        "word_freq_over": 0,
        "word_freq_remove": 0,
        "word_freq_internet": 0,
        "word_freq_order": 0,
        "word_freq_mail": 0,
        "word_freq_receive": 0,
        "word_freq_will": 0,
        "word_freq_people": 0,
        "word_freq_report": 0,
        "word_freq_addresses": 0,
        "word_freq_free": 0,
        "word_freq_business": 0,
        "word_freq_email": 0,
        "word_freq_you": 0,
        "word_freq_credit": 0,
        "word_freq_your": 0,
        "word_freq_font": 0,
        "word_freq_000": 0,
        "word_freq_money": 0,
        "word_freq_hp": 0,
        "word_freq_hpl": 0,
        "word_freq_george": 0,
        "word_freq_650": 0,
        "word_freq_lab": 0,
        "word_freq_labs": 0,
        "word_freq_telnet": 0,
        "word_freq_857": 0,
        "word_freq_data": 0,
        "word_freq_415": 0,
        "word_freq_85": 0,
        "word_freq_technology": 0,
        "word_freq_1999": 0,
        "word_freq_parts": 0,
        "word_freq_pm": 0,
        "word_freq_direct": 0,
        "word_freq_cs": 0,
        "word_freq_meeting": 0,
        "word_freq_original": 0,
        "word_freq_project": 0,
        "word_freq_red": 0,
        "word_freq_edu": 0,
        "word_freq_table": 0,
        "word_freq_conference": 0,
        "char_freq_;": 0,
        "char_freq_(": 0,
        "char_freq_[": 0,
        "char_freq_!": 0,
        "char_freq_$": 0,
        "char_freq_#": 0,
        "capital_run_length_average": 0,
        "capital_run_length_longest": 0,
        "capital_run_length_total": 0,
        "spam": 0
    }

    count_keys = counts.keys()
    max_count = 0
    sequence_lengths = []
    total_uc_count = 0

    try:
        email_file = open(file, "r", encoding = "utf-8")
        for line in email_file:
            print(line)
            words = line.split(" ")
            word_count += len(words)
            char_count += len(line)
            for word in words:
                if word in count_keys:
                    counts[word] += 1
                current_count = 0
                for char in word:
                    if char.isupper():
                        current_count += 1
                        if current_count > max_count:
                            max_count = current_count
                        total_uc_count += 1
                    else:
                        if current_count > 0:
                            sequence_lengths.append(current_count)
                        current_count = 0
                if current_count > 0:
                    sequence_lengths.append(current_count)

    except UnicodeDecodeError as e:
        return None

    email_data["capital_run_length_longest"] = max_count
    email_data["capital_run_length_average"] = calculate_seq_mean(sequence_lengths)
    email_data["capital_run_length_total"] = total_uc_count

    for key in count_keys:
        if key in chars:
            email_data[f"char_freq_{key}"] = (100 * counts[key]) / char_count
        else:
            email_data[f"word_freq_{key}"] = (100 * counts[key]) / word_count

    print(email_data)
    return email_data

def get_index_values(json_data):
    values = []
    for key, value in json_data.items():
        values.append(value)
        #values.append(str(value))
    print(values)
    return values
    #return ",".join(values)

for email in os.listdir(folder):
    # Obtener la ruta completa del archivo
    complete_path = os.path.join(folder, email)
    print(complete_path)
    #file = open(email, "r", encoding = "utf-8")
    json_data = get_email_data(complete_path)
    if json_data is not None:
        with open("Train_enron.csv", mode='a', newline='') as csv_file:
            # Create the CSV writer
            csv_writer = csv.writer(csv_file)
            
            # Convert the JSON to a single line of text
            #csv_line = json.dumps(json_data)
            print(get_index_values(json_data))
            csv_line = get_index_values(json_data)
            
            # Write the line to the CSV file
            csv_writer.writerow(csv_line)

