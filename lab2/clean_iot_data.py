"""
========
Lab 2
========
"""

import pandas as pd
import numpy as np

RAW_FILE = r"E:\angly\iot_sensor_raw_data_extended.csv"
OUT_CLEANED = r"E:\angly\cleaned_iot_sensor_data.csv"
OUT_REPORT = r"E:\angly\data_quality_report.csv"

# ----------------------------------------------------------------------
# กำหนด Data Quality Rules
# ----------------------------------------------------------------------
TEMP_MIN, TEMP_MAX = 0, 50          # อุณหภูมิต้องอยู่ระหว่าง 0-50 องศาเซลเซียส
BATTERY_MIN, BATTERY_MAX = 0, 100    # แบตเตอรี่ต้องอยู่ระหว่าง 0-100

# ----------------------------------------------------------------------
# โหลดข้อมูลดิบ 
# ----------------------------------------------------------------------
raw = pd.read_csv(RAW_FILE)
raw_row_count = len(raw)
print(f"[STEP 1] โหลดข้อมูลดิบสำเร็จ: {raw_row_count} แถว")

df = raw.copy()  # คัดลอกสำรองไว้นะ

# ----------------------------------------------------------------------
# ตรวจจับปัญหาคุณภาพข้อมูลก่อนแก้ไขเพื่อทำรายงาน
# ----------------------------------------------------------------------
issues = pd.DataFrame(index=df.index)
issues["missing_device_id"] = df["device_id"].isna() | (df["device_id"].astype(str).str.strip() == "")
issues["missing_temp"] = df["temp_c"].isna()
issues["missing_motion"] = df["motion"].isna()
issues["missing_battery"] = df["battery"].isna()
issues["temp_out_of_range"] = df["temp_c"].notna() & ((df["temp_c"] < TEMP_MIN) | (df["temp_c"] > TEMP_MAX))
issues["battery_out_of_range"] = df["battery"].notna() & ((df["battery"] < BATTERY_MIN) | (df["battery"] > BATTERY_MAX))
issues["duplicate_row"] = df.duplicated(subset=["device_id", "timestamp"], keep=False)
motion_valid_raw = {"true", "false"}
issues["invalid_motion_format"] = df["motion"].notna() & (~df["motion"].astype(str).str.lower().isin(
    {"true", "false", "0", "1", "yes", "no"}))

summary_before = {
    "จำนวนแถวทั้งหมด (ก่อนทำความสะอาด)": raw_row_count,
    "device_id ว่าง": int(issues["missing_device_id"].sum()),
    "temp_c ขาดหาย (missing)": int(issues["missing_temp"].sum()),
    "motion ขาดหาย (missing)": int(issues["missing_motion"].sum()),
    "battery ขาดหาย (missing)": int(issues["missing_battery"].sum()),
    "temp_c ผิดช่วง (outlier, นอก 0-50)": int(issues["temp_out_of_range"].sum()),
    "battery ผิดช่วง (outlier, นอก 0-100)": int(issues["battery_out_of_range"].sum()),
    "แถวซ้ำ (duplicate device_id+timestamp)": int(issues["duplicate_row"].sum()),
    "motion รูปแบบไม่ถูกต้อง (เช่น TRUEE, unknown)": int(issues["invalid_motion_format"].sum()),
}

print("\n[STEP 2] สรุปปัญหาที่พบ (ก่อนทำความสะอาด):")
for k, v in summary_before.items():
    print(f"  - {k}: {v}")

# ----------------------------------------------------------------------
# STEP 3: ทำความสะอาดข้อมูล
# ----------------------------------------------------------------------

# 3.1 ลบข้อมูลซ้ำ (duplicate rows ที่มี device_id + timestamp เหมือนกันทุกค่า) -> เก็บแถวแรกไว้
before_dedup = len(df)
df = df.drop_duplicates(subset=["device_id", "timestamp"], keep="first").reset_index(drop=True)
removed_duplicates = before_dedup - len(df)
print(f"\n[STEP 3.1] ลบข้อมูลซ้ำ: ลบไป {removed_duplicates} แถว (เหลือ {len(df)} แถว)")

# 3.2 แปลงค่าผิดปกติ (outlier) ให้เป็น NULL (NaN) แทนที่จะลบทั้งแถว
mask_temp_bad = df["temp_c"].notna() & ((df["temp_c"] < TEMP_MIN) | (df["temp_c"] > TEMP_MAX))
mask_batt_bad = df["battery"].notna() & ((df["battery"] < BATTERY_MIN) | (df["battery"] > BATTERY_MAX))
n_temp_nulled = int(mask_temp_bad.sum())
n_batt_nulled = int(mask_batt_bad.sum())
df.loc[mask_temp_bad, "temp_c"] = np.nan
df.loc[mask_batt_bad, "battery"] = np.nan
print(f"[STEP 3.2] แปลงค่าผิดปกติเป็น NULL: temp_c {n_temp_nulled} แถว, battery {n_batt_nulled} แถว")

# 3.3 ทำความสะอาดคอลัมน์ motion ให้เป็นรูปแบบมาตรฐาน (boolean True/False) ค่าที่แปลไม่ได้ -> NaN
motion_map = {
    "true": True, "false": False,
    "1": True, "0": False,
    "yes": True, "no": False,
    "truee": True, "falsee": False,  # แก้คำสะกดผิด
}
def normalize_motion(v):
    if pd.isna(v):
        return np.nan
    key = str(v).strip().lower()
    return motion_map.get(key, np.nan)  # ค่าที่ไม่รู้จัก เช่น "unknown" -> NaN

n_motion_before_unknown = df["motion"].notna().sum()
df["motion"] = df["motion"].apply(normalize_motion)
n_motion_nulled = int((df["motion"].isna()).sum() - (raw["motion"].isna().sum() - removed_duplicates * 0))
print(f"แปลงคอลัมน์ motion เป็น boolean มาตรฐาน (ค่าที่แปลไม่ได้ -> NULL)")

# 3.4 จัดการ Missing Value
# - device_id ว่าง: ไม่สามารถเดาค่าที่ถูกต้องได้ -> คงไว้เป็น NULL และตั้งสถานะเป็น error (ไม่ impute)
# - temp_c ว่าง/เป็น NULL: เติมด้วยค่าเฉลี่ยเคลื่อนที่ (interpolate) แยกตาม device_id เรียงตามเวลา
# - battery ว่าง: เติมด้วยวิธี forward-fill/backward-fill แยกตาม device_id (แบตเตอรี่เปลี่ยนแปลงช้า)
# - motion ว่าง/แปลงไม่ได้: เติมด้วยค่า mode (ค่าที่พบบ่อยที่สุด) ของอุปกรณ์นั้น ๆ

df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

df["temp_c_imputed"] = False
df["battery_imputed"] = False
df["motion_imputed"] = False

for dev_id, grp_idx in df.groupby("device_id", dropna=True).groups.items():
    idx = list(grp_idx)
    # temp_c: interpolate ตามลำดับเวลาของอุปกรณ์นั้น ๆ
    before = df.loc[idx, "temp_c"]
    interpolated = before.interpolate(method="linear", limit_direction="both").round(1) # เพื่มให้ใช้ทศนิยมแค่ 1
    changed = before.isna() & interpolated.notna()
    df.loc[idx, "temp_c_imputed"] = changed
    df.loc[idx, "temp_c"] = interpolated

    # battery: forward-fill แล้ว backward-fill (แบตไม่เปลี่ยนเร็ว)
    before_b = df.loc[idx, "battery"]
    filled = before_b.ffill().bfill()
    changed_b = before_b.isna() & filled.notna()
    df.loc[idx, "battery_imputed"] = changed_b
    df.loc[idx, "battery"] = filled

    # motion: เติมด้วย mode ของอุปกรณ์นั้น
    before_m = df.loc[idx, "motion"]
    if before_m.notna().any():
        mode_val = before_m.mode(dropna=True).iloc[0]
    else:
        mode_val = np.nan
    changed_m = before_m.isna()
    df.loc[idx, "motion_imputed"] = changed_m
    df.loc[idx, "motion"] = before_m.fillna(mode_val)

n_temp_imputed = int(df["temp_c_imputed"].sum())
n_batt_imputed = int(df["battery_imputed"].sum())
n_motion_imputed = int(df["motion_imputed"].sum())
print(f"[STEP 3.4] เติมค่า Missing Value: temp_c {n_temp_imputed} แถว, "
      f"battery {n_batt_imputed} แถว, motion {n_motion_imputed} แถว")
print("           (device_id ที่ว่าง จะไม่ถูกเดาค่า เนื่องจากไม่มีข้อมูลอ้างอิงเพียงพอ)")

# ----------------------------------------------------------------------
# STEP 4: เพิ่มคอลัมน์ data_quality_status
# ----------------------------------------------------------------------
def build_status(row):
    flags = []
    if pd.isna(row["device_id"]) or str(row["device_id"]).strip() == "":
        flags.append("missing_device_id")
    if row["temp_c_imputed"]:
        flags.append("temp_imputed")
    if row["battery_imputed"]:
        flags.append("battery_imputed")
    if row["motion_imputed"]:
        flags.append("motion_imputed")
    if pd.isna(row["temp_c"]):
        flags.append("temp_still_missing")
    if pd.isna(row["battery"]):
        flags.append("battery_still_missing")
    if pd.isna(row["motion"]):
        flags.append("motion_still_missing")
    return "ok" if not flags else "|".join(flags)

df["data_quality_status"] = df.apply(build_status, axis=1)
df = df.drop(columns=["temp_c_imputed", "battery_imputed", "motion_imputed"])

# ----------------------------------------------------------------------
# STEP 5: สรุปผลก่อนหลังและบันทึกไฟล์
# ----------------------------------------------------------------------
after_row_count = len(df)

summary_after = {
    "จำนวนแถวทั้งหมด (หลังทำความสะอาด)": after_row_count,
    "แถวที่ถูกลบ (ซ้ำ)": removed_duplicates,
    "แถวสถานะ ok (ไม่มีปัญหา)": int((df["data_quality_status"] == "ok").sum()),
    "แถวที่ถูกเติมค่า temp_c": n_temp_imputed,
    "แถวที่ถูกเติมค่า battery": n_batt_imputed,
    "แถวที่ถูกเติมค่า motion": n_motion_imputed,
    "แถวที่ device_id ยังว่าง (ต้องตรวจสอบ manual)": int(df["data_quality_status"].str.contains("missing_device_id").sum()),
}

print("\n[STEP 5] สรุปผลหลังทำความสะอาด:")
for k, v in summary_after.items():
    print(f"  - {k}: {v}")

print("\n[STEP 5] เปรียบเทียบจำนวนแถวก่อน-หลัง:")
print(f"  ก่อนทำความสะอาด : {raw_row_count} แถว")
print(f"  หลังทำความสะอาด : {after_row_count} แถว (ลบแถวซ้ำออก {removed_duplicates} แถว)")
print(f"  หมายเหตุ: ไม่มีการลบแถวเนื่องจาก outlier/missing value ออกจากชุดข้อมูล "
      f"— มีเพียงการแปลงเป็น NULL หรือเติมค่า (impute) และระบุสถานะไว้ในคอลัมน์ data_quality_status")

# บันทึกไฟล์
import os
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
df.to_csv(OUT_CLEANED, index=False)

report_rows = []
for k, v in summary_before.items():
    report_rows.append({"ขั้นตอน": "ก่อนทำความสะอาด", "รายการ": k, "จำนวน": v})
for k, v in summary_after.items():
    report_rows.append({"ขั้นตอน": "หลังทำความสะอาด", "รายการ": k, "จำนวน": v})
report_df = pd.DataFrame(report_rows)
report_df.to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")

print(f"\nบันทึกไฟล์ Cleaned Data ที่: {OUT_CLEANED}")
print(f"บันทึกไฟล์รายงานสรุปที่: {OUT_REPORT}")
