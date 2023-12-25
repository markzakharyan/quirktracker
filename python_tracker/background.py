import time
from knappen import *
import pandas as pd
import os


def process(passed: list, inputPath: str, outputPath: str) -> None:
    """
    Process the data from a given CSV file, filtering and extracting relevant information 
    based on the provided list of passed events. The extracted data is then saved to a new CSV file.

    Parameters:
    - passed (list): A list of EventID values that have been pre-approved or filtered. 
                     Only rows in the dataset with an EventID in this list will be processed.
    - inputPath (str): The path to the CSV file containing the data to be processed.
    - outputPath (str): The path to the CSV file to save the processed data to.

    Outputs:
    - A CSV file named "Bgd_500_1jet_wCuts.csv" in the "HitFiles/Bgd" directory, containing the processed data. 
      The columns in the CSV file are "EventID", "TruthID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", 
      "px[GeV]", "py[GeV]", and "pz[GeV]".

    Notes:
    - The function iterates over each particle in the CSV file, filtering based on the provided list of passed events.
    - Rows with problematic RunPoint executions are skipped, and the total number of such problematic executions is printed.
    """
    
    start_time = time.time()

    DD = pd.read_csv(inputPath).to_numpy()

    data = [["EventID", "TruthID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]]
    oopsies=0
    # ii loops over particles in CSV, not over events. About 50 background particles in each event
    for ii in range(1,min(88860, len(DD))):
        TruthID = ii+2
        vecs = [DD[ii][1:5], DD[ii][1:5]]
        PID = DD[ii][0]
        EventID = DD[ii][8]
        
        if EventID not in passed:
            continue

        try:
            AA = RunPoint(vecs[0], vecs[1], 20, False, False)
        except:
            oopsies+=1
            continue


        if len(AA) == 16:
            # jj loops over the 8 layers * 2 particles
            for jj in range(1, 9):
                select = 2 * (jj - 1) + 1 if PID > 0 else 2 * jj
                layer = AA[select - 1][0]
                r = ATLASradiiPixel[layer - 1]
                z = AA[select - 1][1]
                phi = AA[select - 1][2]

                listhere = [EventID, TruthID, PID, layer, r, phi, z] + list(vecs[0])

                if phi.imag == 0 and z.imag == 0:
                    data.append(listhere)

    print(f"There were {oopsies} problem RunPoint executions.")

    
    df = pd.DataFrame(data[1:], columns=data[0])

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    df.to_csv(outputPath, index=False)

    print(f"(Background Regular) Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    # passed list provided here only for debugging purposes
    passed = [3, 4, 10, 14, 15, 19, 20, 21, 26, 31, 32, 33, 40, 41, 43, 47, 50, 51, 52, 55, 56, 57, 58, 59, 61, 66, 67, 68, 69, 70, 79, 81, 82, 83, 87, 88, 94, 95, 98, 100, 103, 104, 105, 111, 113, 114, 116, 117, 118, 119, 127, 128, 131, 134, 136, 139, 140, 141, 142, 143, 144, 150, 152, 154, 155, 165, 166, 171, 173, 174, 176, 177, 178, 180, 181, 184, 186, 187, 188, 189, 192, 194, 195, 197, 198, 199, 203, 205, 208, 211, 212, 220, 222, 224, 228, 229, 234, 236, 237, 240]
    
    current_directory = os.getcwd()
    inputPath = f"{current_directory}/python_tracker/4vector_pionbgd_wCuts.csv"

    outputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles", "Bgd", "Bgd_500_1jet_wCuts.csv")
    
    process(passed, inputPath, outputPath)