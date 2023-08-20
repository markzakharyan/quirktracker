import time
from knappen import *
import pandas as pd
import os
from joblib import Parallel, delayed
from typing import List, Tuple

Lambda = 20

def process_event_batch(event_ids: List[int], DDA: ndarray, DDP: ndarray, Lambda: float) -> Tuple[List[List], List[int]]:
    """
    Process a batch of particle events to generate hit points for each event.

    Parameters:
    - event_ids (List[int]): List of event IDs to process in the current batch.
    - DDA (np.ndarray): Particle data array for particles with PID 15.
    - DDP (np.ndarray): Particle data array for particles with PID -15.
    - Lambda (float): The 'quirk tension' value used for processing.

    Returns:
    - Tuple[List[List], List[int]]:
        - List[List]: Nested list containing data for each hit point. Each inner list contains details of the hit point.
        - List[int]: List of event IDs that passed the processing conditions.

    Notes:
    - This function processes events where the particle ID (PID) is 15 and its antiparticle ID is -15.
    - The function returns hit point details.
    """
    data = []
    passed = []
    for EventID in event_ids:
        vecs = [DDA[EventID], DDP[EventID]]
        AA = RunPoint(vecs[0], vecs[1], Lambda, False, True)
        flag = len(AA) == 16
        if flag:
            passed.append(EventID+1)
            for jj in range(16):
                layer = AA[jj][0] - 1
                r = ATLASradiiPixel[layer]
                z = AA[jj][1]
                phi = AA[jj][2]
                QID = AA[jj][6] - 1
                listhere = [EventID+1, layer+1, QID+1, r, phi, z, vecs[QID][0], vecs[QID][1], vecs[QID][2], vecs[QID][3]]
                data.append(listhere)
    return data, passed


def process() -> List[int]:
    """
    Process particle data from two CSV files, generate hit points for each particle using the RunPoint function, 
    and save the processed data to a new CSV file. 

    Parameters:
    None

    Returns:
    - list: A list of Event IDs that passed the processing conditions.

    Notes:
    - This function processes particle data where the particle ID (PID) is 15 and its antiparticle ID is -15.
    - Only the first 100 events that meet the specified conditions are processed.
    - The output file is saved in the "HitFiles" directory and is named according to the mass and Lambda value.
    """

    pid = 15
    mass = 500

    current_directory = os.getcwd()
    readfileP = f"{current_directory}/4vector_{mass}GeV_PID{pid}_1jet.csv"
    readfileA = f"{current_directory}/4vector_{mass}GeV_PID{-pid}_1jet.csv"

    # Reading data
    DDP = pd.read_csv(readfileP).to_numpy()
    DDA = pd.read_csv(readfileA).to_numpy()

    # keep only first 4 columns of ddp and dda
    DDP = DDP[:, :4]
    DDA = DDA[:, :4]

    num_cores = os.cpu_count()  # Number of CPU cores
    total_events = min(500, len(DDP))
    event_ids = np.arange(total_events)
    
    # Split the event_ids into chunks for parallel processing
    chunks = np.array_split(event_ids, num_cores)
    
    results = Parallel(n_jobs=num_cores)(delayed(process_event_batch)(chunk, DDA, DDP, Lambda) for chunk in chunks)


    # Combine the results from the parallel tasks
    data = [["EventID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]]
    passed = []
    for chunk_data, chunk_passed in results:
        data.extend(chunk_data)
        passed.extend(chunk_passed)
    

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles")

    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"QuirkMass_{mass}_Lambda_{Lambda}_multiprocessed.csv")

    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv(path, index=False)

    return passed

if __name__ == '__main__':
    """
    Main execution of the script.

    The script processes particle data from two input CSV files, generates hit points for each particle event, 
    and saves the processed data to a new CSV file.
    """
    start_time = time.time()
    process()
    print(f"(Quirk Multiprocessed) Time taken: {time.time() - start_time} seconds")