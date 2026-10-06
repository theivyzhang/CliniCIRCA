"""Write dataset/sample_patient.parquet: one fully synthetic patient.

The patient, the note, the IDs and the dates are invented. The columns and
types match the MIMIC-III discharge-summary extract used in the study, so the
pipeline runs on this file exactly as it would on the real data.

    python dataset/make_sample_patient.py
"""

from pathlib import Path

import pandas as pd

NOTE = """\
Admission Date:  [**2162-8-14**]              Discharge Date:   [**2162-8-19**]

Date of Birth:  [**2128-2-11**]             Sex:   F

Service: MEDICINE

Allergies:
Penicillins

Attending:[**First Name3 (LF) 90417**]
Chief Complaint:
Intentional acetaminophen ingestion

Major Surgical or Invasive Procedure:
None

History of Present Illness:
Ms. [**Known lastname 90001**] is a 34 yo female with major depressive
disorder and generalized anxiety disorder who presented to the
[**Hospital1 9001**] Emergency Department on [**2162-8-14**] after an
intentional ingestion of approximately forty 500 mg acetaminophen
tablets. She reports taking the pills around 2 AM, about 6 hours
before arrival, after an argument with her sister. She called a
crisis line at 7 AM and EMS brought her in. She reports two weeks
of worsening low mood, poor sleep, and passive suicidal ideation
after losing her job. Her outpatient psychiatrist increased her
sertraline from 50 mg to 100 mg daily one month prior to
admission. She denies co-ingestion of alcohol or other medications.

In the ED, initial vitals were T 98.6, HR 104, BP 118/76, RR 16,
SpO2 99% on RA. Acetaminophen level was 162 mcg/mL. AST was 45 and
ALT was 52. She was started on IV N-acetylcysteine (NAC) per the
21-hour protocol. Toxicology was consulted and she was admitted to
the MICU for monitoring.

Past Medical History:
1. Major depressive disorder, recurrent
2. Generalized anxiety disorder
3. Prior suicide attempt by overdose in [**2157**], no ICU stay
4. Mild intermittent asthma
5. s/p appendectomy at age 21

Social History:
Lives alone in an apartment. Worked as a graphic designer until
two weeks ago. Smokes 1/2 ppd for 10 years. Drinks alcohol
socially, 2-3 drinks per month. Denies illicit drug use.

Family History:
Mother with bipolar disorder. Father died at age 61 of a
myocardial infarction. No family history of liver disease.

Physical Exam:
On admission:
Vitals: T 98.4, HR 98, BP 116/74, RR 14, SpO2 99% RA
General: Tearful, cooperative, in no acute distress
HEENT: Sclera anicteric, MMM
Neck: Supple, no JVD
CV: Regular rate and rhythm, no murmurs
Lungs: Clear to auscultation bilaterally, no wheezes
Abdomen: Soft, mild RUQ tenderness, no rebound or guarding
Ext: No edema
Neuro: Alert and oriented x3, no asterixis

On discharge:
Vitals: T 98.1, HR 76, BP 112/70, RR 14, SpO2 99% RA
General: Calm, appropriate, in no acute distress
Abdomen: Soft, non-tender
Neuro: Alert and oriented x3, no asterixis

Pertinent Results:
[**2162-8-14**] 08:40AM   ACETAMINOPHEN-162 SALICYLATE-NEG ETHANOL-NEG
[**2162-8-14**] 08:40AM   AST-45 ALT-52 ALK PHOS-71 TOT BILI-0.6
[**2162-8-14**] 08:40AM   PT-12.9 INR(PT)-1.1
[**2162-8-14**] 08:40AM   GLUCOSE-96 UREA N-11 CREAT-0.7 SODIUM-139
POTASSIUM-3.9
[**2162-8-14**] 09:15AM URINE  bnzodzpn-NEG barbitrt-NEG opiates-NEG
cocaine-NEG amphetmn-NEG
[**2162-8-16**] 06:10AM   AST-388 ALT-412 TOT BILI-1.1 INR(PT)-1.4
[**2162-8-19**] 06:05AM   AST-96 ALT-201 TOT BILI-0.7 INR(PT)-1.1

RUQ ultrasound [**2162-8-15**]: Normal liver echotexture. No biliary
dilatation. Patent hepatic vasculature.

Brief Hospital Course:
34 yo female with major depressive disorder admitted after an
intentional acetaminophen overdose.

# Acetaminophen overdose / drug-induced liver injury: NAC was
started in the ED on [**2162-8-14**]. Transaminases rose to a peak
ALT of 412 on [**2162-8-16**] with a mild INR elevation to 1.4, so
NAC was continued beyond the standard 21-hour protocol per
Toxicology. She never developed encephalopathy. She was transferred
from the MICU to the medical floor on [**2162-8-16**]. NAC was stopped
on [**2162-8-17**] once the acetaminophen level was undetectable and
transaminases were down-trending. RUQ ultrasound was unremarkable.

# Suicide attempt / major depressive disorder: She had a 1:1 sitter
throughout the admission. Psychiatry was consulted on hospital day
1 and recommended inpatient psychiatric admission once medically
cleared. Sertraline was held on admission and restarted at 100 mg
daily on [**2162-8-17**]. She engaged well with the psychiatry team
and denied active suicidal ideation at discharge.

# Tobacco use: Nicotine patch was started on admission. She was
counseled on smoking cessation.

# Asthma: No exacerbation. Albuterol was continued as needed.

Medications on Admission:
1. Sertraline 100 mg PO daily
2. Hydroxyzine 25 mg PO Q6H PRN anxiety
3. Albuterol inhaler 2 puffs Q4H PRN wheezing

Discharge Medications:
1. Sertraline 100 mg PO daily
2. Nicotine 14 mg/24 hr patch, apply one patch daily
3. Albuterol inhaler 2 puffs Q4H PRN wheezing

Discharge Disposition:
Extended Care

Facility:
[**Hospital 9002**] Inpatient Psychiatry

Discharge Diagnosis:
Primary:
Intentional acetaminophen overdose
Drug-induced liver injury

Secondary:
Major depressive disorder, recurrent, severe
Generalized anxiety disorder
Tobacco use

Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.

Discharge Instructions:
You were admitted after taking too much acetaminophen (Tylenol).
You were treated with an antidote called N-acetylcysteine and your
liver tests are improving. You are being transferred to an
inpatient psychiatry unit for further care.

Please do not take any acetaminophen (Tylenol) until cleared by
your doctor. Hydroxyzine was stopped.

Followup Instructions:
Please have your liver function tests checked in one week.
Follow up with your primary care provider, Dr. [**Last Name (STitle) 90533**],
on [**2162-8-28**] at 10:30 AM.
"""

# ICD-9 codes in MIMIC format (no dots). Psychiatric codes are those in 290-319.
ALL_ICD9 = ["9654", "E9500", "5733", "29633", "30002", "3051", "49390"]
PSYCH_ICD9 = [c for c in ALL_ICD9 if c[:3].isdigit() and 290 <= int(c[:3]) <= 319]

PATIENT = {
    "HADM_ID": 990001.0,
    "SUBJECT_ID": 900001,
    "GENDER": "F",
    "DOB": "2128-02-11",
    "ADMITTIME": "2162-08-14",
    "DISCHTIME": "2162-08-19",
    "ADMISSION_TYPE": "EMERGENCY",
    "RELIGION": "NOT SPECIFIED",
    "MARITAL_STATUS": "SINGLE",
    "ETHNICITY": "UNKNOWN/NOT SPECIFIED",
    "AGE_AT_CHARTDATE": 34,
    "CHARTDATE": "2162-08-19",
    "TEXT": NOTE,
    "ALL_ICD9_CODES": "; ".join(ALL_ICD9),
    "PSYCH_ICD9_CODES": "; ".join(PSYCH_ICD9),
    "NUM_ICD9_CODES": float(len(ALL_ICD9)),
    "NUM_PSYCH_ICD9_CODES": float(len(PSYCH_ICD9)),
    "DS_COMPONENTS": "Report",
    "NOTE_LENGTH": len(NOTE),
}

# Same dtypes as the study data (IDs and code counts are floats there).
DTYPES = {
    "HADM_ID": "float64", "SUBJECT_ID": "int64", "AGE_AT_CHARTDATE": "int64",
    "NUM_ICD9_CODES": "float64", "NUM_PSYCH_ICD9_CODES": "float64", "NOTE_LENGTH": "int64",
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_patient.parquet"
    pd.DataFrame([PATIENT]).astype(DTYPES).to_parquet(out, index=False)
    print(f"Wrote {out}")
