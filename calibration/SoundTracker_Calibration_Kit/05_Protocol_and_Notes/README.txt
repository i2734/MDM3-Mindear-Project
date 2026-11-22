SOUNDTRACKER CALIBRATION — FILE PROCEDURE

1. Open Calibration_Trial_Register.xlsx > Trial Register.
2. Before each trial, find its preassigned Trial ID.
3. Write down reference LAeq and any start/end timestamps; if the meter cannot export, manually enter reference LAeq.
4. Export the phone app data. Rename its raw CSV to the exact Expected filename. Save it in the corresponding Folder path.
5. Mark Raw saved? = Yes and QC status = Recorded or Checked.
6. Keep raw exports unmodified. Store meter exports in 04_Reference_Meter_Exports; log meter file/time mapping in Reference Log.
7. Process the one-second phone LAeq values into one 15-second LAeq per trial; do not average dB arithmetically.
8. Fill Phone LAeq and calculate Error = Phone minus Reference.

E1_80dB is the shared 80 dB, 0-degree, unobstructed baseline for E2 and E3. Do NOT record duplicate baseline trials.
IP01–IP03 and SA01–SA03 are device IDs; record actual models and contributor buckets in Device Register.
The same 15-second analysis window should be used for every trial.

If app exports contain several trials in one file, preserve the original file in the raw folder and create separately named trial extracts in 02_Processed_Data. Record the source export name and timestamps in the register notes.
