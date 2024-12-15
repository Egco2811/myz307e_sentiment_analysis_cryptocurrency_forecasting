import csv
import os
import shutil

def annotate_csv(csv_filepath):
    temp_filepath = csv_filepath + ".temp"
    if os.path.exists(temp_filepath):
        resume_from_temp = True
    else:
        resume_from_temp = False

    with open(csv_filepath, 'r', newline='', encoding='utf-8') as infile, \
         open(temp_filepath, 'a', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not resume_from_temp:
            if 'sentiment' not in fieldnames:
                fieldnames.append('sentiment')
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

        for row in reader:
            if 'sentiment' not in row or not row['sentiment']:
                if 'text' in row:
                    print(row['text'])
                    while True:
                        sentiment = input("1. Positive\n2. Negative\n3. Neutral\nType 'end' to finish current batch.\nType the number: ")
                        if sentiment.lower() == 'end':
                            return
                        try:
                            int(sentiment)
                            row['sentiment'] = sentiment
                            if not resume_from_temp:
                                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                            writer.writerow(row)
                            break
                        except ValueError:
                            print("Invalid input. Please enter 1, 2, 3, or 'end'.")
                else:
                    print("Text column not found in the row. The row will be included with NaN sentiment.")
                    row['sentiment'] = float('nan')



def merge_files(csv_filepath, temp_filepath):
    merged_filepath = csv_filepath + ".merged"

    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as infile, \
             open(temp_filepath, 'r', newline='', encoding='utf-8') as temp_file, \
             open(merged_filepath, 'w', newline='', encoding='utf-8') as merged_file:

            reader = csv.DictReader(infile)
            temp_reader = csv.DictReader(temp_file)
            fieldnames = reader.fieldnames
            if 'sentiment' not in fieldnames:
                fieldnames.append('sentiment')

            merged_writer = csv.DictWriter(merged_file, fieldnames=fieldnames)
            merged_writer.writeheader()

            temp_rows = list(temp_reader)
            temp_index = 0

            for row in reader:
                if 'sentiment' not in row or not row['sentiment']:
                    if temp_index < len(temp_rows):
                        merged_writer.writerow(temp_rows[temp_index])
                        temp_index += 1
                    else:
                        merged_writer.writerow(row)

                else:
                    merged_writer.writerow(row)




    except PermissionError:
        print(f"Error: Could not access or modify {csv_filepath}. Please make sure it is not open in another program.")
        input("Press Enter to exit...")
        exit()

    try:
        shutil.move(merged_filepath, csv_filepath)
        os.remove(temp_filepath)
    except OSError as e:
        print(f"Error merging or deleting files: {e}")
        input("Press Enter to exit...")
        exit()



csv_file_path = "processed_tweets.csv"

while True:
    annotate_csv(csv_file_path)
    merge_files(csv_file_path, csv_file_path + ".temp")
    another_batch = input("Annotate another batch? (yes/no): ")
    if another_batch.lower() != 'yes':
        break