import time
from knappen import *
import pandas as pd
import os

def process(Lambda, inputPathP, inputPathA, outputPath) -> list:
    """
    Process particle data from two CSV files, generate hit points for each particle using the RunPoint function, 
    and save the processed data to a new CSV file. 

    Parameters:
    - Lambda (float): The 'quirk tension' value used for processing.
    - inputPathP (str): The path to the CSV file containing the data for particles with PID 15.
    - inputPathA (str): The path to the CSV file containing the data for particles with PID -15.
    - outputPath (str): The path to the CSV file to save the processed data to.

    Returns:
    - list: A list of Event IDs that passed the processing conditions.

    Notes:
    - This function processes particle data where the particle ID (PID) is 15 and its antiparticle ID is -15.
    - Only the first 100 events that meet the specified conditions are processed.
    - The output file is saved in the "HitFiles" directory and is named according to the mass and Lambda value.
    """

    

    # Reading data
    DDP = pd.read_csv(inputPathP).to_numpy()
    DDA = pd.read_csv(inputPathA).to_numpy()

    # keep only first 4 columns of ddp and dda
    DDP = DDP[:, :4]
    DDA = DDA[:, :4]


    # Initialize data table and other variables
    data = [["EventID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]]
    passed = []
    count = 1

    for EventID in range(min(500, len(DDP))):
        vecs = [DDA[EventID], DDP[EventID]]
        AA = RunPoint(vecs[0], vecs[1], Lambda, False, True)

        if len(AA) == 16:
            
            passed.append(EventID+1)
            
            for jj in range(16):
                layer = AA[jj][0] - 1
                r = ATLASradiiPixel[layer]
                z = AA[jj][1]
                phi = AA[jj][2]
                QID = AA[jj][6] - 1
                listhere = [EventID+1, layer+1, QID+1, r, phi, z, vecs[QID][0], vecs[QID][1], vecs[QID][2], vecs[QID][3]] # ensure the +1s are correct???
                data.append(listhere)
                count += 1


    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv(outputPath, index=False)

    return passed

if __name__ == '__main__':
    """
    Main execution of the script.

    The script processes particle data from two input CSV files, generates hit points for each particle event, 
    and saves the processed data to a new CSV file.
    """

    pid = 15
    mass = 500
    Lambda = 20


    current_directory = os.getcwd()

    inputPathP = f"{current_directory}/4vector_{mass}GeV_PID{pid}_1jet.csv"
    inputPathA = f"{current_directory}/4vector_{mass}GeV_PID{-pid}_1jet.csv"

    outputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles")
    os.makedirs(outputPath, exist_ok=True)
    outputPath = os.path.join(outputPath, f"QuirkMass_{mass}_Lambda_{Lambda}.csv")

    
    start_time = time.time()
    passed = process(Lambda, inputPathP, inputPathA, outputPath)

    print(f"(Quirk Regular) Time taken: {time.time() - start_time} seconds")
    print(f"passed: {passed}")