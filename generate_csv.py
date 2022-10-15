import csv
import pandas as pd

fhandle = open('sample.csv', 'r')


def writeCSV(List):
    with open("sample.csv", mode="r", newline='') as old_file:
        read_content = csv.reader(old_file)

        with open("sample.csv", mode="a", newline='') as new_file:
            write_content = csv.writer(new_file, delimiter=',')
            write_content.writerow(List)