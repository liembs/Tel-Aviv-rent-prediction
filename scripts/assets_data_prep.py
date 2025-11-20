import pandas as pd
import numpy as np


#!/usr/bin/env python
# coding: utf-8

# <div dir="rtl" lang="he">
# 
# <h1 align="center">🏠 ניתוח וחיזוי נתוני דירות להשכרה בתל אביב</h1>
# 
# <hr>
# 
# <h3 align="center">🔍 קורס: כרייה וניתוח נתונים מתקדם בפייתון</h3>
# 
# <h3 align="center">💻 מגישות העבודה:</h3>
# 
# <p align="center">
#   <table align="center" style="border: none;">
#     <thead>
#       <tr>
#         <th>👩‍💻 שם המגישה</th>
#         <th>🆔 תעודת זהות</th>
#       </tr>
#     </thead>
#     <tbody>
#       <tr>
#         <td>מור היימן</td>
#         <td>322466418</td>
#       </tr>
#       <tr>
#         <td>ליאם בן שושן</td>
#         <td>211467576</td>
#       </tr>
#     </tbody>
#   </table>
# </p>
# 
# <br>
# 
# <h4 align="center">🔗 קישור לגיט:</h4>
# 
# <p align="center">
#   <a href="https://github.com/mor2800/tel-aviv-rent-analysis.git">
#     https://github.com/mor2800/tel-aviv-rent-analysis.git
#   </a>
# </p>
# 
# </div>
# 

# <div dir="rtl">
# 
# #### 🔍 חלק ראשון – פונקציות ולוגיקה לניקוי וסידור הנתונים
# 
# </div>
# 



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import re
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# קריאת קובץ CSV
# df = pd.read_csv("train.csv") 


# In[2]:


# 1️⃣ הסרת ערכים חסרים
# df = df.dropna(subset=['price'])

# 2️⃣ הסרת דירות למכירה (לא רלוונטיות להשכרה)
#if 'description' in df.columns:
 #   df = df[~df['description'].str.contains('למכירה', case=False, na=False)]

# 3️⃣ סינון טווחי מחיר סביר
# df = df[(df['price'] >= 800) & (df['price'] <= 40000)]


# In[3]:


def clean_property_type(df, verbose=True):
    """
    פונקציה לניקוי וניתוח עמודת 'property_type' (כולל תרגום מרוסית וסינון ערכים לא רלוונטיים).
    """
    df_clean = df.copy()

    df_clean = df_clean.dropna(subset=['property_type'])

    # תרגום מרוסית
    df_clean['property_type'] = df_clean['property_type'].replace('Квартира', 'דירה')

    # תיקון תקלות typo
    df_clean['property_type'] = df_clean['property_type'].replace('דירת גן להשכרה', 'דירת גן')
    df_clean['property_type'] = df_clean['property_type'].replace('גג/ פנטהאוז', 'גג/פנטהאוז')
    df_clean['property_type'] = df_clean['property_type'].replace('גג/פנטהאוז להשכרה', 'גג/פנטהאוז')
    df_clean['property_type'] = df_clean['property_type'].replace('דירה להשכרה', 'דירה')
    # ערכים לא רצויים
    not_allowed_property_types = [
        'באתר מופיע ערך שלא ברשימה הסגורה',
        'מרתף/פרטר',
        'חניה',
        'מחסן',
        'כללי',
        'החלפת דירות',
        'סאבלט'
    ]
    
    # סינון
    before = len(df_clean)
    df_clean = df_clean[~df_clean['property_type'].isin(not_allowed_property_types)]
    df_clean = df_clean[df_clean['property_type'].notna()]
    after = len(df_clean)
    
    if verbose:
        print(f"📌 סינון שורות לפי שורות שמייצגות דירה בלבד (ולא מחסן וכדומה)")
        print(f"📌 שורות שנמחקו: {before - after}")
    
      
    
    return df_clean
   


# In[4]:


def fill_missing_room_numbers(df):

    df.loc[
    df['description'].str.contains('להשכרה !! ברחוב טהון - דיזנגוף, 4 חדרים משופצת קומה 2 ללא מעלית כ 90 מ״ר משופצת', na=False), 
    'room_num'
] = 4

    """
    פונקציה שמעדכנת את עמודת 'room_num' לפי טקסט בעמודת 'description' במידה וערך 'room_num' הוא 0.
    """
    for idx, row in df[df['room_num'] == 0].iterrows():
        description = str(row['description']).lower()

        # 1️⃣ מספר לפני 'חדר' (למשל 3 חדרים)
        match = re.search(r'(\d+(?:\.\d+)?)\s*חדר', description)
        if match:
            extracted_room_num = float(match.group(1))
            df.at[idx, 'room_num'] = extracted_room_num

        # 2️⃣ מספר לפני 'חד' (למשל 3 חד')
        elif re.search(r'(\d+(?:\.\d+)?)\s*חד', description):
            match = re.search(r'(\d+(?:\.\d+)?)\s*חד', description)
            extracted_room_num = float(match.group(1))
            df.at[idx, 'room_num'] = extracted_room_num

        # 3️⃣ מופיע 'חדר וחצי'
        elif 'חדר וחצי' in description:
            df.at[idx, 'room_num'] = 1.5

        # 4️⃣ מופיע 'דירת חדר'
        elif 'דירת חדר' in description:
            df.at[idx, 'room_num'] = 1

        # אחרת — ניקח את החציון לפי 'AREA'
        else:
            area = row.get('AREA', None)
            if area is not None:
                median_room_num = df.loc[
                    (df['AREA'] == area) & (df['room_num'] > 0),
                    'room_num'
                ].median()
                if pd.notna(median_room_num):
                    df.at[idx, 'room_num'] = median_room_num
                else:
                    # fallback אם אין חציון זמין
                    df.at[idx, 'room_num'] = df.loc[df['room_num'] > 0, 'room_num'].median()
            else:
                # fallback אם AREA חסר
                df.at[idx, 'room_num'] = df.loc[df['room_num'] > 0, 'room_num'].median()
    
    print(f"📌עודכנו מספרי החדרים שהיו בערך 0 על פי התיאור או על פי חציון לפי שטח")
    
    
    return df


# In[5]:


def fix_floor_and_total_floors(df):
    """
    פונקציה שמבצעת:
    1️⃣ פיצול עמודת 'floor' שמכילה ערכים בסגנון 'קומה מתוך'
    2️⃣ המרה לערכים מספריים מסוג Int64
    3️⃣ חיפוש ערכי 'floor' חסרים בתיאור והשלמתם
    """
    # 1️⃣ פיצול הערכים בעמודת 'floor' שמכילים 'מתוך'
    for idx, val in df['floor'].items():
        if pd.notna(val) and 'מתוך' in str(val):
            parts = str(val).split('מתוך')
            floor_val = parts[0].strip().replace('קרקע', '0')
            total_floors_val = parts[1].strip().replace('קרקע', '0')
            df.at[idx, 'floor'] = floor_val
            df.at[idx, 'total_floors'] = total_floors_val

    # 2️⃣ המרה לערכים מספריים עם Int64
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce').astype('Int64')
    df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce').astype('Int64')


    # 4️⃣ המרה נוספת ל-Int64
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce').astype('Int64')
    
    print(f"📌תוקנו מספרי הקומות שהיו כוללים מלל")
    return df


# In[6]:


def fill_floors_with_stats(df, stat_choice='median'):
    """
    פונקציה משולבת שמטפלת גם בערכי 'floor' וגם בערכי 'total_floors' בדאטהפריים.

    שלבי עבודה:
    1️⃣ מחשבת סטטיסטיקה (mean/median/mode) לכל שכונה ולכל הדאטה.
       stat_choice:
           סוג הסטטיסטיקה להשלמת ערכים — 'mean', 'median' או 'mode'.
    2️⃣ מתקנת ערכים חריגים ב-'floor':
        - floor חסר (NaN)
        - floor גדול מ-50
        - floor גדול מ-total_floors
        - total_floors חסר (NaN)
      לפני מילוי סטטיסטי — מנסה לחלץ ערך מ-description.
    3️⃣ מתקנת ערכים חריגים ב-'total_floors':
        - total_floors חסר (NaN)
        - total_floors גדול מ-50
        - total_floors קטן מ-floor
        - שורות ששונו ב-floor
    4️⃣ ממירה את הערכים לערכים מספריים מסוג Int64 (כדי לאפשר NaN).
    """
    # 1️⃣ חישוב סטטיסטיקה לפי שכונה
    if stat_choice == 'mean':
        floor_stats = df.groupby('neighborhood')['floor'].mean()
        total_floors_stats = df.groupby('neighborhood')['total_floors'].mean()
        overall_floor_stat = df['floor'].mean()
        overall_total_floors_stat = df['total_floors'].mean()
    elif stat_choice == 'median':
        floor_stats = df.groupby('neighborhood')['floor'].median()
        total_floors_stats = df.groupby('neighborhood')['total_floors'].median()
        overall_floor_stat = df['floor'].median()
        overall_total_floors_stat = df['total_floors'].median()
    elif stat_choice == 'mode':
        floor_stats = df.groupby('neighborhood')['floor'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        total_floors_stats = df.groupby('neighborhood')['total_floors'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        overall_floor_stat = df['floor'].mode().iloc[0] if not df['floor'].mode().empty else None
        overall_total_floors_stat = df['total_floors'].mode().iloc[0] if not df['total_floors'].mode().empty else None
    else:
        raise ValueError("Invalid stat_choice! Use 'mean', 'median' or 'mode'.")

    fixed_rows_floor = []
    fixed_rows_total_floors = []

    # 2️⃣ טיפול ב-'floor'
    for idx, row in df.iterrows():
        floor_val = row['floor']
        total_floors_val = row.get('total_floors', pd.NA)
        description = str(row.get('description', '')).lower()

        invalid_floor = (
            pd.notna(floor_val) and
            pd.notna(total_floors_val) and
            floor_val > total_floors_val
        )
        total_floors_missing = pd.isna(total_floors_val)

        if pd.isna(floor_val) or floor_val > 50 or invalid_floor or total_floors_missing:
            # 2️⃣.א️⃣ מנסה לחלץ ערך מה-description
            extracted_floor = None
            # אופציה 1: "קומה" עם נקודתיים, רווח או מקף ואחריו מספר
            match = re.search(r'קומה[:\s\-]*([0-9]+)', description)
            if match:
                extracted_floor = int(match.group(1))

            # אופציה 2: מופיע "קרקע"
            elif 'קרקע' in description:
                extracted_floor = 0
            if match:
                extracted_floor = int(match.group(1))
            if extracted_floor is not None and extracted_floor <= 50:
                df.at[idx, 'floor'] = extracted_floor
            else:
                # אם לא מצאנו ב-description — ממשיכים לשיטה הקיימת
                neighborhood = row['neighborhood']
                stat_value = floor_stats.get(neighborhood, None)
                if pd.notna(stat_value) and stat_value < 50:
                    df.at[idx, 'floor'] = int(round(stat_value))
                else:
                    df.at[idx, 'floor'] = int(round(overall_floor_stat))
            fixed_rows_floor.append(idx)

    # המרה ל-Int64
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce').astype('Int64')

    # 3️⃣ טיפול ב-'total_floors'
    for idx, row in df.iterrows():
        total_floors_val = row['total_floors']
        floor_val = row['floor']

        invalid_total_floors = (
            pd.notna(total_floors_val) and
            pd.notna(floor_val) and
            total_floors_val < floor_val
        )
        floor_was_fixed = idx in fixed_rows_floor

        if pd.isna(total_floors_val) or total_floors_val > 50 or invalid_total_floors or floor_was_fixed:
            neighborhood = row['neighborhood']
            stat_value = total_floors_stats.get(neighborhood, None)
            if pd.notna(stat_value) and stat_value >= floor_val and stat_value <= 50:
                df.at[idx, 'total_floors'] = int(round(stat_value))
            elif pd.notna(overall_total_floors_stat) and overall_total_floors_stat >= floor_val and overall_total_floors_stat <= 50:
                df.at[idx, 'total_floors'] = int(round(overall_total_floors_stat))
            elif pd.notna(floor_val):
                df.at[idx, 'total_floors'] = int(floor_val) + 1
            fixed_rows_total_floors.append(idx)

    # המרה ל-Int64
    df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce').astype('Int64')

    print(f"✅ מילוי ערכי הקומות הסתיים בהצלחה לפי {stat_choice}!")


    return df


# In[7]:


def tax_fill_zero(df):
    """
    פונקציה שמעדכנת את building_tax ל-0:
    1️⃣ עבור property_type שבהם כל הערכים הם 0 או NaN.
    2️⃣ עבור שורות שבהן total_floors הוא 0 או 1 ויש NaN.
    """
    changed_rows = []

    # 1️⃣ עבור property_type שכל הערכים בו 0 או NaN
    fully_missing = df.groupby('property_type')['building_tax'].apply(
        lambda x: ((x.isna()) | (x == 0)).all()
    )
    fully_missing = fully_missing[fully_missing]

    for prop_type in fully_missing.index:
        affected_rows = df.loc[
            (df['property_type'] == prop_type) & (df['building_tax'].isna())
        ].index
        df.loc[affected_rows, 'building_tax'] = 0
        changed_rows.extend(affected_rows)

    # 2️⃣ עבור total_floors = 0 או 1 עם building_tax = NaN
    affected_rows_tf = df.loc[
        ((df['total_floors'] == 0) | (df['total_floors'] == 1)) & (df['building_tax'].isna())
    ].index
    df.loc[affected_rows_tf, 'building_tax'] = 0
    changed_rows.extend(affected_rows_tf)

    print(f"✅ עודכנו {len(set(changed_rows))} רשומות ל-building_tax = 0 "
          f"({len(fully_missing)} קטגוריות property_type).")
    return df


# In[8]:


def fill_by_address(df_copy):
    updated_rows = []
    mode_address = df_copy[
        (df_copy['building_tax'].notna()) & (df_copy['building_tax'] > 0)
    ].groupby('address')['building_tax'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    for idx, row in df_copy[df_copy['building_tax'].isna()].iterrows():
        address = row['address']
        mode_val = mode_address.get(address, None)
        if pd.notna(mode_val):
            df_copy.at[idx, 'building_tax'] = mode_val
            updated_rows.append(idx)
    return updated_rows, 'כתובת מלאה'

def fill_by_street_floor_elevator(df_copy):
    updated_rows = []
    df_copy['street'] = df_copy['address'].astype(str).apply(
        lambda x: x.strip().split()[0] if pd.notna(x) and len(x.strip().split()) > 0 else None
    )
    mode = df_copy[
        (df_copy['building_tax'].notna()) & (df_copy['building_tax'] > 0)
    ].groupby(['street', 'total_floors', 'elevator'])['building_tax'].median()

    for idx, row in df_copy[df_copy['building_tax'].isna()].iterrows():
        key = (row['street'], row['total_floors'], row['elevator'])
        mode_val = mode.get(key, None)
        if pd.notna(mode_val):
            df_copy.at[idx, 'building_tax'] = mode_val
            updated_rows.append(idx)
    return updated_rows, 'רחוב + קומה + מעלית'

def fill_by_street_elevator(df_copy):
    updated_rows = []
    mode = df_copy[
        (df_copy['building_tax'].notna()) & (df_copy['building_tax'] > 0)
    ].groupby(['street', 'elevator'])['building_tax'].median()

    for idx, row in df_copy[df_copy['building_tax'].isna()].iterrows():
        key = (row['street'], row['elevator'])
        mode_val = mode.get(key, None)
        if pd.notna(mode_val):
            df_copy.at[idx, 'building_tax'] = mode_val
            updated_rows.append(idx)
    return updated_rows, 'רחוב + מעלית'

def fill_by_neigh_floor_elevator(df_copy):
    updated_rows = []
    mode = df_copy[
        (df_copy['building_tax'].notna()) & (df_copy['building_tax'] > 0)
    ].groupby(['neighborhood', 'total_floors', 'elevator'])['building_tax'].median()

    for idx, row in df_copy[df_copy['building_tax'].isna()].iterrows():
        key = (row['neighborhood'], row['total_floors'], row['elevator'])
        mode_val = mode.get(key, None)
        if pd.notna(mode_val):
            df_copy.at[idx, 'building_tax'] = mode_val
            updated_rows.append(idx)
    return updated_rows, 'שכונה + קומה + מעלית'

def fill_by_neigh_elevator(df_copy):
    updated_rows = []
    mode = df_copy[
        (df_copy['building_tax'].notna()) & (df_copy['building_tax'] > 0)
    ].groupby(['neighborhood', 'elevator'])['building_tax'].median()

    for idx, row in df_copy[df_copy['building_tax'].isna()].iterrows():
        key = (row['neighborhood'], row['elevator'])
        mode_val = mode.get(key, None)
        if pd.notna(mode_val):
            df_copy.at[idx, 'building_tax'] = mode_val
            updated_rows.append(idx)
    return updated_rows, 'שכונה + מעלית'

def fill_by_neigh(df_copy):
    updated_rows = []
    mode = df_copy[
        (df_copy['building_tax'].notna()) & (df_copy['building_tax'] > 0)
    ].groupby(['neighborhood'])['building_tax'].median()

    for idx, row in df_copy[df_copy['building_tax'].isna()].iterrows():
        key = row['neighborhood']
        mode_val = mode.get(key, None)
        if pd.notna(mode_val):
            df_copy.at[idx, 'building_tax'] = mode_val
            updated_rows.append(idx)
    return updated_rows, 'שכונה כללית'

def fill_building_tax_advanced(df):
    df_copy = df.copy()
    df_copy['building_tax'] = pd.to_numeric(df_copy['building_tax'], errors='coerce')

    changed_rows = []
    update_counters = {}

    fill_functions = [
        fill_by_address,
        fill_by_street_floor_elevator,
        fill_by_street_elevator,
        fill_by_neigh_floor_elevator,
        fill_by_neigh_elevator,
        fill_by_neigh
    ]

    # מריצים את כל הפונקציות בסדר ההיררכי
    for func in fill_functions:
        updated, label = func(df_copy)
        changed_rows.extend(updated)
        update_counters[label] = len(updated)

    # שלב סופי: כל הערכים שעדיין NaN — נעדכן ל-0
    final_missing = df_copy[df_copy['building_tax'].isna()].index
    df_copy.loc[final_missing, 'building_tax'] = 0
    changed_rows.extend(final_missing)
    update_counters['Defaulted to 0'] = len(final_missing)


    # הדפסת סיכום
    print(f"✅ מילוי building_tax הסתיים! עודכנו {len(set(changed_rows))} רשומות בסך הכל.")

    # עדכון הערכים בדאטה המקורי
    df['building_tax'] = df_copy['building_tax']
    return df


# In[9]:


#נתוני המרחקים נראים מאוד לא מסודרים ולכן אעדיף לטפל בהם מחדש

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🗝️ כאן תכניסי את ה-API Key שלך
#API_KEY = "הכנסי כאן את ה-API Key שלך"


def compute_distance(address):
    """
    פונקציה שמקבלת כתובת ומחזירה את המרחק מכיכר דיזנגוף (במטרים).
    """
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    origin = f"{address}, תל אביב יפו"
    destination = "כיכר דיזנגוף, תל אביב"

    params = {
        "origins": origin,
        "destinations": destination,
        "key": API_KEY,
        "mode": "driving",
        "language": "he"
    }

    try:
        response = requests.get(base_url, params=params)
        result = response.json()
        distance = result["rows"][0]["elements"][0]["distance"]["value"]
        return distance
    except Exception as e:
        print(f"שגיאה: {e} בכתובת: {address}")
        return None

def update_distance_from_address(df, compute_distance_func, max_workers=5):
    """
    פונקציה שמעדכנת את העמודה 'distance_from_center' עבור כל כתובת
    לפי חישוב מרחק שניתן בפונקציה compute_distance_func.
    בנוסף:
    🔹 מסירה ערכים חריגים מעל 50,000 מטר.
    🔹 ממירה את הערך ממטרים לקילומטרים.
    🔹 אם נשארים ערכים חסרים — משלימה לפי ממוצע השכונה.
    """
    df_copy = df.copy()

    # שלב 1️⃣ — הפעלת ThreadPoolExecutor לקריאות API במקביל
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, row in df_copy.iterrows():
            address = row['address']
            if pd.notna(address):
                futures[executor.submit(compute_distance_func, address)] = idx

        for future in as_completed(futures):
            idx = futures[future]
            distance = future.result()
            if distance is not None:
                df_copy.at[idx, 'distance_from_center'] = distance

    # שלב 2️⃣ — סימון ערכים חריגים (מעל 50,000 מטר) כ-NaN
    df_copy.loc[df_copy['distance_from_center'] > 50000, 'distance_from_center'] = pd.NA

    # שלב 3️⃣ — המרת הערכים ממטרים לקילומטרים
    df_copy['distance_from_center'] = df_copy['distance_from_center'] / 1000

    # שלב 4️⃣ — מילוי ערכים חסרים לפי ממוצע השכונה
    for neighborhood, group in df_copy.groupby('neighborhood'):
        median_distance = group['distance_from_center'].mean()
        df_copy.loc[group.index, 'distance_from_center'] = df_copy.loc[group.index, 'distance_from_center'].fillna(median_distance)

    print("✅ המרחקים עודכנו בהצלחה! (כולל טיפול בערכים קיימים וחסרים)")
    return df_copy


# In[10]:


import re

def fill_area_by_room_num(df):
    """
    פונקציה שמתקנת ערכים בעייתיים בעמודת 'area':
    🔹 אם הדירה 5 חדרים ומטה והשטח מעל 500 — מחלקים ב-10.
    🔹 מחליפה ערכים חסרים (NaN), קטנים מ-20 או גדולים מ-500.
    🔹 מנסה קודם לחלץ ערך מתוך ה-description (מ"ר או מטר).
    🔹 אם לא נמצא — משלימה לפי ממוצע השטח לכל room_num.
    """
    df_copy = df.copy()

    # מחשבים ממוצע שטח לכל room_num (נעשה פעם אחת)
    area_by_room = df_copy.groupby('room_num')['area'].mean()

    # טיפול בערכים בעייתיים
    for idx, row in df_copy.iterrows():
        area_val = row['area']
        room_num = row['room_num']
        description = str(row.get('description', '')).lower()

        # 🔹 טיפול מיוחד: אם 5 חדרים ומטה והשטח מעל 500 — מחלקים ב-10
        if pd.notna(area_val) and room_num <= 5 and area_val > 500:
            df_copy.at[idx, 'area'] = area_val / 10
            continue

        # 🔹 טיפול בערכים בעייתיים (NaN, קטן מ-20, גדול מ-500)
        if pd.isna(area_val) or area_val < 20 or area_val > 500:
            extracted_area = None

            # מנסה לחפש מתוך ה-description (כולל "מ"ר" או "מטר")
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:מ"ר|מטר)', description)
            if match:
                extracted_area = float(match.group(1))
                if 20 <= extracted_area <= 500:
                    df_copy.at[idx, 'area'] = extracted_area
                    continue  # מצאנו ערך סביר — לא צריך להמשיך הלאה

            # אם לא מצאנו בתיאור — נלך על הממוצע
            avg_area = area_by_room.get(room_num, df_copy['area'].mean())
            df_copy.at[idx, 'area'] = avg_area

    print("✅ כל הערכים בעמודת 'area' תוקנו בהצלחה לפי room_num ותיאור!")
    return df_copy


# In[11]:


def fill_monthly_arnona_by_area(df):
    """
    פונקציה לחישוב monthly_arnona לפי מחיר ממוצע למ"ר.
    1️⃣ מתקנת ידנית כתובת ספציפית.
    2️⃣ מחשבת מחיר ממוצע למ"ר על סמך כל הנתונים הקיימים.
    3️⃣ ממלאת ערכים חסרים (NaN), נמוכים מ-50 או גבוהים מ-8000 לפי חישוב: area * מחיר ממוצע למ"ר.
    """
    df_copy = df.copy()

    # 1️⃣ תיקון ידני
    df_copy.loc[
        (df_copy['address'] == 'יפת 203') & 
        (df_copy['room_num'] == 2.0) & 
        (df_copy['area'] == 24.0), 
        'monthly_arnona'
    ] = 170

    # 2️⃣ חישוב מחיר ממוצע למ"ר
    valid_data = df_copy[
        (df_copy['monthly_arnona'].notna()) & 
        (df_copy['area'].notna()) & 
        (df_copy['area'] > 0)
    ]
    avg_arnona_per_meter = (valid_data['monthly_arnona'] / valid_data['area']).mean()


    # 3️⃣ מילוי ערכים בעייתיים
    mask = (
        (df_copy['monthly_arnona'].isna()) | 
        (df_copy['monthly_arnona'] < 50) | 
        (df_copy['monthly_arnona'] > 4000)
    )
    df_copy.loc[mask, 'monthly_arnona'] = df_copy.loc[mask, 'area'] * avg_arnona_per_meter

    print("✅ מילוי monthly_arnona לפי מחיר ממוצע למ\"ר הסתיים בהצלחה!")
    return df_copy


# In[12]:



from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

def get_target_encoder_mapping_with_names(df_train, target_col='price'):
    """
    פונקציה שמייצרת Target Encoding לשכונה (neighborhood) ומחזירה מילון עם שמות השכונות.
    
    df_train : pandas.DataFrame
        הדאטהפריים עם הנתונים.
    target_col : str
        שם עמודת היעד (למשל 'price').
    
    מחזירה:
    -------
    mapping_dict : dict
        מילון Target Encoding עם שמות השכונות.
    """
    df_train = df_train.copy()
    
    # שלב 1️⃣ - Label Encoding (כדי שיהיה מספר לכל שכונה)
    le = LabelEncoder()
    df_train['neighborhood_encoded'] = le.fit_transform(df_train['neighborhood'])
    
    # שלב 2️⃣ - Target Encoding
    encoder = TargetEncoder(cols=['neighborhood_encoded'])
    df_train['neighborhood_encoded_te'] = encoder.fit_transform(
        df_train['neighborhood_encoded'], df_train[target_col]
    )
    
    # שלב 3️⃣ - בניית מילון עם שמות השכונות
    mapping_series = encoder.mapping['neighborhood_encoded']
    
    if isinstance(mapping_series, pd.Series):
        mapping_dict = mapping_series.to_dict()
    elif isinstance(mapping_series, pd.DataFrame):
        mapping_dict = dict(zip(mapping_series.iloc[:, 0], mapping_series.iloc[:, 1]))
    else:
        raise ValueError("❌ mapping אינו נתמך.")
    
    # המרה חזרה לשמות השכונות
    reverse_map = dict(zip(df_train['neighborhood_encoded'], df_train['neighborhood']))
    mapping_dict_named = {reverse_map.get(k, k): v for k, v in mapping_dict.items()}
    if 64 in mapping_dict_named:
        value = mapping_dict_named.pop(64)
        mapping_dict_named['אזורי חן'] = value

    print("✅ Target Encoder Mapping נוצר בהצלחה עם שמות שכונות!")
    return mapping_dict_named
# mapping_dict = get_target_encoder_mapping_with_names(df)


# In[13]:


def map_neighborhood_using_dict_from_target_encoder(df):

    #הכנסה ידנית של מילון לאחר החישוב על מנת להמנע בבעיה בטסט"
    mapping_dict= {'אפקה': 8582.929441232816,
    'בית שטראוס': 9681.241821221134,
    'בצרון': 8002.711732315873,
    'גבעת הרצל': 10947.83717839303,
    'גני צהלה': 8495.09820303163,
    'דקר': 7942.0908229559645,
    'הגוש הגדול': 8360.064889980895,
    'הדר יוסף': 8632.493900203212,
    'המשתלה': 10035.957494314862,
    'הצפון החדש החלק הדרומי': 8611.759302118968,
    'הצפון החדש החלק הצפוני': 10053.758712025301,
    'הצפון החדש סביבת ככר המדינה': 8470.811595894615,
    'הצפון הישן החלק הדרום מזרחי': 8655.48956030745,
    'הצפון הישן החלק הדרום מערבי': 9087.991436485998,
    'הצפון הישן החלק המרכזי': 8174.653050322661,
    'הצפון הישן החלק הצפוני': 8716.32323253632,
    'הקריה': 9595.787010649172,
    'התקווה': 8787.97033413318,
    'חוף הצוק': 8651.903311999233,
    'יד אליהו': 9108.167103738224,
    'יפו ג': 8701.786316335469,
    'יפו ד': 7928.455590579053,
    'כוכב הצפון': 8859.502580417657,
    'כפיר': 8212.883830585051,
    'כרם התימנים': 9896.490193821402,
    'לב תל אביב החלק הדרומי': 8647.05657724418,
    'לב תל אביב החלק הצפוני': 8485.740082320417,
    'לבנה': 9701.221381954394,
    'לינקולן': 8735.382639308684,
    'מונטיפיורי': 9004.8774739849,
    'מע"ר צפוני': 10243.246073316755,
    'מרכז יפו מזרחית לשדרות ירושלים': 9132.308470569087,
    'נאות אפקה א': 8865.967205811807,
    'נאות אפקה ב': 10093.465361040622,
    'נוה אביבים': 9072.775263045552,
    'נוה אליעזר': 8592.59826667252,
    'נוה ברבור': 8449.813894036355,
    'נוה חן': 9344.56179965995,
    'נוה שאנן': 9042.286477433172,
    'נוה שרת': 10436.801035950484,
    'נחלת יצחק': 12005.752029946549,
    'נמל תל אביב': 9044.664880984501,
    "עג'מי": 9379.535639532223,
    'עזרא': 8995.946897908112,
    'פארק צמרת': 8725.240154039358,
    'פלורנטין': 8264.014034806514,
    'צהלון': 9044.664880984501,
    'צפון יפו': 8320.964619072967,
    'קרית שלום': 8394.122509169512,
    'רביבים': 8916.739091655016,
    'רמת אביב': 8196.357628137755,
    'רמת אביב ג': 9403.685765032109,
    'רמת אביב החדשה': 10345.294580980297,
    'רמת החייל': 9562.663538588356,
    'רמת הטייסים': 8381.677547563151,
    'רמת ישראל': 9149.996789963669,
    'שבזי': 8929.114233970973,
    'שיכון בבלי': 9395.888618931747,
    'שיכוני חסכון': 7725.273973754477,
    'שפירא': 9566.360960061997,
    'תל ברוך צפון': 7816.840446236782,
    'תל חיים': 8629.717444980046,
    'תל כביר': 10384.628386198954,
    'אזורי חן': 8570.43785724551}


    df = df.copy()
    df['neighborhood'] = df['neighborhood'].map(mapping_dict).fillna(8700)
    print("✅ עמודת 'neighborhood' הוחלפה בהצלחה לפי המילון!")
    return df


# df
# <div dir="rtl">
# 
# ### 🔍prepare_data
# #####    פונקציה מאחדת לניקוי וסידור הנתונים   
# 
# </div>
# 

# In[14]:


def prepare_data(df, mode):
    """
    פונקציה זו מבצעת עיבוד מקדים לנתוני הדירות.
    df: DataFrame עם נתוני הגלם.
    mode: 'train' או 'test' — קובע אם לבצע ניקוי חריגים.
    הפונקציה:
    - מנקה נתונים חריגים (רק ב-train)
    - מסירה עמודות מיותרות
    - משלימה ערכים חסרים
    - ממירה משתנים קטגוריאליים ל-One-Hot Encoding
    הפלט: DataFrame מוכן לאימון/חיזוי.
    """

    # 1️⃣ ניקוי נתונים חריגים ב-train
    if mode == 'train' and 'price' in df.columns:
        # הסרת ערכים חסרים
        df = df.dropna(subset=['price'])
        # הסרת דירות למכירה (לא רלוונטי להשכרה)
        df = df[~df['description'].str.contains('למכירה', case=False, na=False)]
        # סינון טווחי מחיר לא סבירים
        df = df[(df['price'] >= 800) & (df['price'] <= 40000)]

    # 2️⃣ הסרת דירות למכירה גם ב-test (כי הן לא רלוונטיות)
    if 'description' in df.columns:
        df = df[~df['description'].str.contains('למכירה', case=False, na=False)]

    # 3️⃣ הרצת פונקציות ניקוי נתונים
    df = fill_missing_room_numbers(df)
    df = clean_property_type(df)
    df = fill_missing_room_numbers(df)
    df = fix_floor_and_total_floors(df)
    df = fill_floors_with_stats(df, stat_choice='median')
    df = tax_fill_zero(df)
    df = fill_building_tax_advanced(df)
    df = fill_area_by_room_num(df)
    df = fill_monthly_arnona_by_area(df)
    df = map_neighborhood_using_dict_from_target_encoder(df)

    #(כנראה שאין חצר) השלמת ערכים חסרים בגודל חצר ל-0
    df['garden_area'] = np.where(df['garden_area'] > 0, 1, 0)


    # 4️⃣ הסרת עמודות מיותרות
    drop_cols = ['address', 'description', 'days_to_enter', 'num_of_payments','num_of_images','distance_from_center']
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)



    # 5️⃣ המרת משתנים קטגוריאליים ל-One-Hot Encoding
    if 'property_type' in df.columns:
        df = pd.get_dummies(df, columns=['property_type'], drop_first=True)
    

    # 6️⃣ סידור אלפביתי של העמודות
    df = df.reindex(sorted(df.columns), axis=1)
    
    #למקרה חירום
    df.fillna(0, inplace=True)


    return df



# In[15]:


# import numpy as np
# df = pd.read_csv("train.csv") 
# df= prepare_data(df, "train")
