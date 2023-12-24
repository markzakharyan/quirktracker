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
        print(f"EventId: {EventID}")
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

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
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

    inputPathP = f"{current_directory}/python_tracker/4vector_{mass}GeV_PID{pid}_1jet.csv"
    inputPathA = f"{current_directory}/python_tracker/4vector_{mass}GeV_PID{-pid}_1jet.csv"

    outputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles", f"QuirkMass_{mass}_Lambda_{Lambda}.csv")

    
    start_time = time.time()
    passed = process(Lambda, inputPathP, inputPathA, outputPath)

    print(f"(Quirk Regular) Time taken: {time.time() - start_time} seconds")
    print(f"passed: {passed}")


# (Quirk Regular) Time taken: 2937.8659579753876 seconds
# passed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 387, 388, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500]