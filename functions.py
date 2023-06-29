import pandas as pd

def featurize_datetime_index(df, daytime=True):
    df = df.copy()

    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    df["weekday_name"] = df.index.strftime("%A")
    df["month"] = df.index.month
    df["month_name"] = df.index.strftime("%B")
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["week_of_year"] = df.index.weekofyear
    df["day_of_year"] = df.index.dayofyear

    if daytime:
        # Add column with category for time of day:
        # midnight, early_morning, late_morning, afternoon, evening, night
        def time_of_day(hour):
            if hour >= 0 and hour < 6:
                return "midnight"
            elif hour >= 6 and hour < 9:
                return "early_morning"
            elif hour >= 9 and hour < 12:
                return "late_morning"
            elif hour >= 12 and hour < 15:
                return "afternoon"
            elif hour >= 15 and hour < 18:
                return "evening"
            else:
                return "night"

        df["time_of_day"] = (df["hour"].apply(time_of_day)).astype("category")

    df["weekday_name"] = df["weekday_name"].astype("category")
    df["month_name"] = df["month_name"].astype("category")

    return df
