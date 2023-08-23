import os
import time
import pandas as pd
from knappen import *
from joblib import Parallel, delayed



def process_chunk(chunk: list, passed: list, DD: ndarray) -> Tuple[list, int]:
    """
    Process a chunk of the data, generating the required data list and counting any errors.

    Parameters:
    - chunk (list): A list of indices representing rows of the data to be processed.
    - passed (list): A list of EventID values
    - DD (ndarray): The data to be processed.

    Returns:
    - list: A list containing the processed data.
    - int: Count of errors encountered during processing.
    """

    data = []
    oopsies = 0
    for ii in chunk:
        TruthID = ii + 2
        vecs = [DD[ii][1:5], DD[ii][1:5]]
        PID = DD[ii][0]
        EventID = DD[ii][8]

        if EventID not in passed:
            continue


        try:
            AA = RunPoint(vecs[0], vecs[1], 20, False, False)
        except:
            oopsies += 1
            continue


        if len(AA) == 16:
            for jj in range(1, 9):
                select = 2 * (jj - 1) + 1 if PID > 0 else 2 * jj
                layer = AA[select - 1][0]
                r = ATLASradiiPixel[layer - 1]
                z = AA[select - 1][1]
                phi = AA[select - 1][2]

                listhere = [EventID, TruthID, PID, layer, r, phi, z] + list(vecs[0])

                if phi.imag == 0 and z.imag == 0:
                    data.append(listhere)

    return data, oopsies

def process(passed: list, inputPath: str, outputPath: str) -> None:
    """
    Processes a dataset in parallel, filtering and extracting relevant information 
    based on the provided list of passed events. The extracted data is then saved 
    to a CSV file.

    Parameters:
    - passed (list): A list of EventID values that have been pre-approved or filtered. 
                     Only rows in the dataset with an EventID in this list will be processed.
    - inputPath (str): The path to the CSV file containing the data to be processed.
    - outputPath (str): The path to the CSV file to save the processed data to.

    Outputs:
    - A CSV file named "Bgd_500_1jet_wCuts_multiprocessed.csv" in the "HitFiles/Bgd" directory, 
      containing the processed data. The columns in the CSV file are "EventID", "TruthID", "PID", 
      "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", and "pz[GeV]".

    Notes:
    - The function utilizes all available CPU cores to process the data in parallel.
    - After processing, the function prints the total number of problematic executions 
      (represented by the "oopsies" count) encountered during the parallel processing.
    """
    start_time = time.time()

    
    DD = pd.read_csv(inputPath).to_numpy()
    
    indices = list(range(1, min(88860, len(DD))))
    chunk_size = len(indices) // os.cpu_count()
    chunks = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]

    results = Parallel(n_jobs=os.cpu_count())(delayed(process_chunk)(chunk, passed, DD) for chunk in chunks)


    # Combine results from all processes
    all_data = [["EventID", "TruthID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]]
    total_oopsies = 0
    for data_chunk, oopsies_chunk in results:
        all_data.extend(data_chunk)
        total_oopsies += oopsies_chunk

    print(f"There were {total_oopsies} problem RunPoint executions.")

    
    df = pd.DataFrame(all_data[1:], columns=all_data[0])

    os.makedirs(os.path.join(*outputPath.split(os.path.sep)[:-1]), exist_ok=True)
    df.to_csv(outputPath, index=False)

    print(f"(Background Multiprocessed) Time taken: {time.time() - start_time} seconds")





if __name__ == "__main__":
    """
    Main execution of the script.
    
    The script processes the data in parallel using multiple cores. 
    The results from each process are combined and saved to a CSV file.
    """

    passed = [3, 4, 10, 14, 15, 19, 20, 21, 26, 31, 32, 33, 40, 41, 43, 47, 50, 51, 52, 55, 56, 57, 58, 59, 61, 66, 67, 68, 69, 70, 79, 81, 82, 83, 87, 88, 94, 95, 98, 100, 103, 104, 105, 111, 113, 114, 116, 117, 118, 119, 127, 128, 131, 134, 136, 139, 140, 141, 142, 143, 144, 150, 152, 154, 155, 165, 166, 171, 173, 174, 176, 177, 178, 180, 181, 184, 186, 187, 188, 189, 192, 194, 195, 197, 198, 199, 203, 205, 208, 211, 212, 220, 222, 224, 228, 229, 234, 236, 237, 240]


    current_directory = os.getcwd()
    inputPath = f"{current_directory}/4vector_pionbgd_wCuts.csv"

    outputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles", "Bgd", "Bgd_500_1jet_wCuts_multiprocessed.csv")

    process(passed, inputPath, outputPath)

    