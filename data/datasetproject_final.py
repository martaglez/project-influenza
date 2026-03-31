import pandas as pd
import os

base_dir = os.path.dirname(__file__)

input_path = os.path.join(base_dir, "sentinelTestsDetectionsPositivity.csv")

data = pd.read_csv(input_path)

flu = data[
    data["pathogen"].str.contains("Influenza", case=False, na=False) &
    (data["indicator"].str.lower() == "detections") &
    (data["age"].str.lower().isin(["total", "all"])) &
    (data["pathogensubtype"].str.lower() == "total")
].copy()

flu = flu[["countryname", "yearweek", "value"]].rename(columns = {"countryname": "country", "value": "cases"})

#yearweek --> date (first date of the week)
flu["date"] = pd.to_datetime(flu["yearweek"] + "-1", format = "%Y-W%W-%w", errors = "coerce")

flu["year"] = flu["date"].dt.year
flu["week"] = flu["date"].dt.isocalendar().week

flu_weekly = flu.groupby(["country", "year", "week"], as_index = False)["cases"].sum()

flu_weekly = flu_weekly.sort_values(["country","year","week"]).reset_index(drop = True)

output_path = os.path.join(base_dir, "influenza_clean_weekly.csv")

flu_weekly.to_csv(output_path, index = False)

print(flu_weekly.head())
print(f"Rows in final dataset: {len(flu_weekly)}")
print("CSV saved as 'influenza_clean_weekly.csv'")