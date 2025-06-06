from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

"""
Data object for processing Biomark raw files.

Requires data for "Passive Reference ROX". Otherwise does not work.
"""
class ChipType(Enum):
    s192_a24 = "192.24"
    s96_a96 = "96.96"

def get_assay_groups(file_path: Optional[str], chip_type: ChipType) -> dict:
    """
    Processes assay grouping .csv file. 
    """
    # If file_path is None, we just assume we have no assay replicates
    num_assays = int(chip_type.value.split(".")[1])
    # if chip_type is ChipType.s96_a96: num_assays = 24
    if file_path is None:
        return {"R1": list(range(1, num_assays + 1))}

    groups = defaultdict(list)
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            row_group = line[1]
            row_ind = int(line[0])
            groups[row_group].append(row_ind)

    # Make sure that the total number of wells assigned to groups equals the number of assay wells
    total_assigned = 0
    for g in groups.items():
        total_assigned += len(g[1])

    assert total_assigned == num_assays, \
        f"Total number of wells in assay replicate file ({total_assigned}) should equal total number of wells for specified chip type ({num_assays})."

    return groups

class Biomark:
    # Instantiate class with file location
    def __init__(
        self, 
        file_location: str, 
        chip_type: ChipType,
        probe_name: str = "SNPtype-FAM",
    ):
        # Lists for raw data and header info
        rox_raw = []
        fam_raw = []
        rox_bkgd = []
        fam_bkgd = []
        headers = []

        self.num_samples = int(chip_type.value.split(".")[0])
        self.num_assays = int(chip_type.value.split(".")[1])

        # 96.96 format has SXX format
        self.sample_pad = 2 if self.num_samples == 96 else 3
        # Always set to 2
        self.assay_pad = 2

        # Open file
        with open(file_location) as csv_file:
            # Initiate CSV reader
            csv_reader = csv.reader(csv_file, delimiter=',')

            # Most recent data category
            cats = [
                "Raw Data for Passive Reference ROX",
                "Raw Data for Probe " + probe_name,
                "Bkgd Data for Passive Reference ROX",
                "Bkgd Data for Probe " + probe_name,
            ]
            data_agg = defaultdict(list)
            headers = None

            # Temp vars
            curr_cat = None
            lines_processed = 0

            # Number of lines to read per probe section
            line_limit = self.num_assays * self.num_samples

            # Loop through CSV file
            for row in csv_reader:
                if row and row[0] == "Cycle Number":
                    temp_row = row.copy()
                    temp_row[0] = "Chamber ID"
                    headers = list(filter(None, temp_row))
                # If you run into a new section of data, set curr_cat and lines_processed
                if row and row[0] != curr_cat and row[0] in cats:
                    curr_cat = row[0]
                    lines_processed = 0
                    # After setting, move onto the next row
                    continue
                if curr_cat != None and row and row[0] != "Chamber ID" and lines_processed < line_limit:
                    data_agg[curr_cat].append(row[:len(headers)])
                    # Increment number of lines processed and continue
                    lines_processed += 1
                    continue
                # Reset if we finish a section
                if curr_cat != None and lines_processed >= line_limit:
                    curr_cat = None
                    lines_processed = 0

        # Close opened csv file
        csv_file.close()

        # Temp vars
        rox_raw = data_agg["Raw Data for Passive Reference ROX"]
        fam_raw = data_agg["Raw Data for Probe " + probe_name]
        rox_bkgd = data_agg["Bkgd Data for Passive Reference ROX"]
        fam_bkgd = data_agg["Bkgd Data for Probe " + probe_name]

        # Dataframe from raw data
        df_rox_raw = pd.DataFrame(rox_raw, columns=headers)
        df_fam_raw = pd.DataFrame(fam_raw, columns=headers)
        df_rox_bkgd = pd.DataFrame(rox_bkgd, columns=headers)
        df_fam_bkgd = pd.DataFrame(fam_bkgd, columns=headers)

        # Get column names
        cols = df_fam_raw.columns.drop('Chamber ID')

        # Converts all values to numeric values
        df_rox_raw[cols] = df_rox_raw[cols].apply(pd.to_numeric)
        df_fam_raw[cols] = df_fam_raw[cols].apply(pd.to_numeric)
        df_rox_bkgd[cols] = df_rox_bkgd[cols].apply(pd.to_numeric)
        df_fam_bkgd[cols] = df_fam_bkgd[cols].apply(pd.to_numeric)

        # Subtracts background data from FAM and ROX data (background from Biomark)
        df_fam_sub = df_fam_raw.copy()
        df_fam_sub[cols] = df_fam_raw[cols].sub(df_fam_bkgd[cols])
        df_rox_sub = df_rox_raw.copy()
        df_rox_sub[cols] = df_rox_raw[cols].sub(df_rox_bkgd[cols])

        # Normalize FAM values with ROX values
        df_fam_rox = df_fam_raw.copy()
        df_fam_rox[cols] = df_fam_sub[cols].divide(df_rox_sub[cols])

        # Split
        df_fam_rox[["Sample ID", "Assay ID"]] = df_fam_rox["Chamber ID"].str.split("-", expand=True)
        df_rox_sub[["Sample ID", "Assay ID"]] = df_rox_sub["Chamber ID"].str.split("-", expand=True)
        df_rox_bkgd[["Sample ID", "Assay ID"]] = df_rox_bkgd["Chamber ID"].str.split("-", expand=True)
        df_rox_raw[["Sample ID", "Assay ID"]] = df_rox_raw["Chamber ID"].str.split("-", expand=True)
        df_fam_bkgd[["Sample ID", "Assay ID"]] = df_fam_bkgd["Chamber ID"].str.split("-", expand=True)
        df_fam_raw[["Sample ID", "Assay ID"]] = df_fam_raw["Chamber ID"].str.split("-", expand=True)

        self.rox_raw = df_rox_raw.copy()
        self.rox_bkgd = df_rox_bkgd.copy()
        self.fam_raw = df_fam_raw.copy()
        self.fam_bkgd = df_fam_bkgd.copy()
        self.fam_rox = df_fam_rox.copy()

    # Subtracts ROX background from ROX raw
    def get_rox(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        # After setting IDs, get the normalized fam_rox values and returns only numerics
        rox_raw_vals = self.rox_raw[
            (self.rox_raw["Assay ID"] == assay_id) &
            (self.rox_raw["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        rox_bkgd_vals = self.rox_bkgd[
            (self.rox_bkgd["Assay ID"] == assay_id) &
            (self.rox_bkgd["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return rox_raw_vals - rox_bkgd_vals
    
    # Subtracts ROX background from ROX raw
    def get_rox_raw(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        # After setting IDs, get the normalized fam_rox values and returns only numerics
        rox_raw_vals = self.rox_raw[
            (self.rox_raw["Assay ID"] == assay_id) &
            (self.rox_raw["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return rox_raw_vals
    
    # Subtracts ROX background from ROX raw
    def get_rox_bkgd(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        rox_bkgd_vals = self.rox_bkgd[
            (self.rox_bkgd["Assay ID"] == assay_id) &
            (self.rox_bkgd["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return rox_bkgd_vals
    
    # Subtracts ROX background from ROX raw
    def get_fam(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        # After setting IDs, get the normalized fam_rox values and returns only numerics
        fam_raw_vals = self.fam_raw[
            (self.fam_raw["Assay ID"] == assay_id) &
            (self.fam_raw["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        fam_bkgd_vals = self.fam_bkgd[
            (self.fam_bkgd["Assay ID"] == assay_id) &
            (self.fam_bkgd["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return fam_raw_vals - fam_bkgd_vals
    
    # Subtracts ROX background from ROX raw
    def get_fam_raw(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        # After setting IDs, get the normalized fam_rox values and returns only numerics
        fam_raw_vals = self.fam_raw[
            (self.fam_raw["Assay ID"] == assay_id) &
            (self.fam_raw["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return fam_raw_vals
    
    # Subtracts ROX background from ROX raw
    def get_fam_bkgd(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        fam_bkgd_vals = self.fam_bkgd[
            (self.fam_bkgd["Assay ID"] == assay_id) &
            (self.fam_bkgd["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return fam_bkgd_vals

    # Given sample and assay well numbers, it returns the time-series data for that well only.
    def get_fam_rox(self, sample, assay):
        # Set sample and assay IDs
        sample_id = "S" + str(sample).zfill(self.sample_pad)
        assay_id = "A" + str(assay).zfill(self.assay_pad)

        # After setting IDs, get the normalized fam_rox values and returns only numerics
        norm_vals = self.fam_rox[
            (self.fam_rox["Assay ID"] == assay_id) &
            (self.fam_rox["Sample ID"] == sample_id)
        ].select_dtypes(include=np.number).values[0]

        return norm_vals