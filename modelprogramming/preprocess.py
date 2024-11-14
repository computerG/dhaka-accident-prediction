import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def classify_car(df_cleaned):
    df_cleaned = df_cleaned
    # Combine the 'Header' and 'News title' columns for cause analysis
    df_cleaned["AccidentDescription"] = (
        df_cleaned["Header"] + "  " + df_cleaned["News title"]
    )
    # Initialize a CountVectorizer to extract common terms (unigrams and bigrams) that may indicate car type
    vectorizer = CountVectorizer(
        max_features=20, ngram_range=(1, 2), stop_words="english"
    )
    X_text = vectorizer.fit_transform(df_cleaned["AccidentDescription"])

    # Summing up the counts of each feature word
    word_counts = X_text.sum(axis=0)
    word_counts_df = pd.DataFrame(
        {"Term": vectorizer.get_feature_names_out(), "Count": word_counts.A1}
    ).sort_values(by="Count", ascending=False)

    # Analyze common vehicle types in accident descriptions by searching for specific keywords
    vehicle_keywords = [
        "bus",
        "truck",
        "car",
        "motorcycle",
        "bike",
        "rickshaw",
        "train",
        "tractor",
        "van",
    ]

    # Create a dictionary to store counts of each vehicle type
    vehicle_counts = {vehicle: 0 for vehicle in vehicle_keywords}

    # Count occurrences of each vehicle type keyword in the descriptions
    for vehicle in vehicle_keywords:
        vehicle_counts[vehicle] = (
            df_cleaned["AccidentDescription"]
            .str.contains(vehicle, case=False, na=False)
            .sum()
        )
    return df_cleaned


def extract_weekdays(input_df):
    df_cleaned = input_df
    df_cleaned["Weekday"] = df_cleaned["Accident date"].dt.dayofweek

    # Map the weekday numbers to names for clarity
    weekday_mapping = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df_cleaned["Weekday"] = df_cleaned["Weekday"].map(weekday_mapping)
    return df_cleaned


# Define a list of Dhaka-area locations and neighborhoods for a detailed search
dhaka_neighborhoods = [
    "Dhanmondi",
    "Gulshan",
    "Banani",
    "Uttara",
    "Mirpur",
    "Mohammadpur",
    "Bashundhara",
    "Badda",
    "Farmgate",
    "Shahbagh",
    "Khilgaon",
    "Mugda",
    "Jatrabari",
    "Malibagh",
    "Dhaka",
    "Gazipur",
    "Bogra",
    "Comilla",
    "Narsingdi",
    "Tangail",
    "Naogaon",
    "Bhola",
    "Brahmanbaria",
    "Rajshahi",
    "Mohakhali",
    "Tejgaon",
    "Rampura",
    "Shyamoli",
    "Keraniganj",
    "Muktagacha",
    "Bangabandhu",
    "Bangladesh",
    "Manikganj",
    "Dinajpur",
    "Ctg",
    "Sylhet",
    "Chittagong",
    "JU Chhatra",
    "Natore",
    "Pabna",
    "Tarikat",
    "Canada",
    "Sirajganj",
    "Oman",
    "Egypt",
    "Faridpur",
    "medicine Road",
    "Jessore",
    "Narail One",
    "Ashulia",
    "Chapainawabganj",
    "Rangpur",
    "California",
    "Narayanganj",
    "Khulna road",
    "Mymensingh",
    "Munshiganj",
    "Gaibandha",
    "Habiganj",
    "Chandpur",
    "Australia",
    "Kushtia",
    "New York",
    "Jhenaidah",
    "Demra",
    "Nepal",
    "Panchagarh",
    "Barisal",
    "Meherpur",
    "Kurmitola",
    "Rajbari",
    "Kurigram",
    "Thakurgaon",
    "Gopalganj",
    "Khulna",
    "Magura road",
    "Uganda",
    "Teknaf",
    "Bandarban",
    "Ratna",
    "Banglamotor",
    "Country road",
    "Pirojpur",
    "Paltan",
    "Noakhali",
    "Sherpur",
    "Lamonirhat",
    "Macedonia",
    "Bangla",
    "Gulistan",
    "Madaripur",
    "Lakshmipur",
    "Atiqul",
    "Rangamati",
    "Panthapath",
    "Barguna",
    "Airport Road",
    "Satkhira",
    "Jhenidah",
    "PM’s office Road",
    "Netrokona",
    "Hatirjheel Lake",
    "Central African Republic",
    "Bogura",
    "Moulvibazar",
    "Chuadanga",
    "Nilphamari",
    "Cox’s Bazar",
    "Kaliakoir",
    "Khagrachhari",
    "Sunamganj",
    "Saudi Arabia",
    "Hatirjheel",
    "Indian Kashmir",
    "Jamalpur",
    "Lalmonirhat",
    "Bagerhat",
    "Utah",
    "Joypurhat",
]


# Function to identify the primary location mentioned in each row
def extract_location_from_text(text):
    for neighborhood in dhaka_neighborhoods:
        if re.search(r"\b" + re.escape(neighborhood) + r"\b", text, re.IGNORECASE):
            return neighborhood
    return "Unknown"


def classify_multiple_car_types(text):
    text = text.lower()
    types = []
    if "bus" in text:
        types.append("Bus")
    if "truck" in text:
        types.append("Truck")
    if "car" in text:
        types.append("Car")
    if "motorcycle" in text:
        types.append("Motorcycle")
    if "motorcyclist" in text:
        types.append("Motorcycle")
    if "bike" in text:
        types.append("Bike")
    if "rickshaw" in text:
        types.append("Rickshaw")
    if "train" in text:
        types.append("Train")
    if "van" in text:
        types.append("Train")

    # Join multiple types if present, otherwise return the single type or 'Other'
    return " & ".join(types) if types else "Other"


# Defining a function to categorize accident types
def classify_accident_type(text):
    text = text.lower()
    if "collision" in text:
        return "Collision"
    elif "pedestrian" in text:
        return "Pedestrian"
    elif "overturn" in text or "overturned" in text:
        return "Overturn"
    elif "hit-and-run" in text or "hit and run" in text:
        return "Hit-and-Run"
    elif "killed" in text or "injured" in text:
        return "Casualty"
    else:
        return "Other"


# Defining severity levels based on 'Deaths' and 'Injured' counts
def classify_severity(deaths, injured):
    if deaths >= 3:
        return "High"
    elif deaths >= 1 or injured >= 3:
        return "Moderate"
    elif deaths == 0 and injured > 0:
        return "Low"
    else:
        return "Minor"


# Apply the function to create 'Severity' column
def fill_numeric_with_median(df):
    numeric_cols = df.select_dtypes(include=np.number).columns  # Select numeric columns
    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: x.fillna(x.median()), axis=0
    )  # Fill NaNs with median
    return df


def preprocess_input_data(data):
    # Convert input JSON to DataFrame
    # Concatenate headers and titles for a comprehensive location frequency analysis
    input_df = data
    text = " ".join(
        input_df["Header"].fillna("") + "  " + input_df["News title"].fillna("")
    )
    median_deaths = input_df["Deaths"].median()
    median_deaths
    input_df.fillna({"Deaths": median_deaths}, inplace=True)
    input_df["Deaths"].astype(int)
    if "Unnamed: 0" in input_df.columns:
        input_df.drop("Unnamed: 0", axis=1, inplace=True)

    # Handle missing values, date conversions, and any other steps
    input_df["Accident date"] = pd.to_datetime(
        input_df["Accident date"], errors="coerce"
    )

    # Drop rows with invalid date formats (if any) after conversion
    input_df.dropna(subset=["Accident date"], inplace=True)

    # Extract year, month, and day from the 'Accident date' for trend analysis
    input_df["Year"] = input_df["Accident date"].dt.year
    input_df["Month"] = input_df["Accident date"].dt.month
    input_df["Day"] = input_df["Accident date"].dt.day
    input_df["Month-Year"] = input_df["Accident date"].dt.to_period("M")
    # Apply the season classification to the data
    input_df["Season"] = input_df["Month"].apply(get_season)
    # Return preprocessed data ready for prediction
    input_df = classify_car(input_df)
    input_df = extract_weekdays(input_df)
    # Create 'Identified Location' column based on extracted locations
    input_df["Identified Location"] = input_df["AccidentDescription"].apply(
        extract_location_from_text
    )
    input_df["Severity"] = input_df.apply(
        lambda x: classify_severity(x["Deaths"], x["Injured"]), axis=1
    )
    # Apply the function to create 'Accident Type' column
    input_df["Accident Type"] = input_df["AccidentDescription"].apply(
        classify_accident_type
    )
    input_df["Car Type"] = input_df["AccidentDescription"].apply(
        classify_multiple_car_types
    )
    input_df["Deaths"] = input_df["Deaths"].astype("int64")
    input_df.to_csv("Cleaneddf.csv")
    print(input_df.shape)
    return input_df
