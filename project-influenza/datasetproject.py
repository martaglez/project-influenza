import pandas as pd

file_path = "/Users/marta/Downloads/sentinelTestsDetectionsPositivity.csv"
data = pd.read_csv(file_path)

flu = data[data["pathogen"].str.contains("Influenza", case = False, na = False) &
    (data["indicator"].str.lower() == "detections") &
    (data["age"].str.lower().isin(["total", "all"]))].copy()

flu = flu[["countryname", "yearweek", "value"]].rename(columns = {"countryname": "country", "value": "cases"})

#yearweek --> date (first date of the week)
flu["date"] = pd.to_datetime(flu["yearweek"] + "-1", format = "%Y-W%W-%w", errors = "coerce")

flu["year"] = flu["date"].dt.year
flu["week"] = flu["date"].dt.isocalendar().week

flu_weekly = flu.groupby(["country", "year", "week"], as_index = False)["cases"].sum()

flu_weekly = flu_weekly.sort_values(["country","year","week"]).reset_index(drop = True)

flu_weekly.to_csv("influenza_clean_weekly.csv", index = False)

print(flu_weekly.head())
print(f"Rows in final dataset: {len(flu_weekly)}")
print("CSV saved as 'influenza_clean_weekly.csv'")