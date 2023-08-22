import quirk_multiprocessing, background_multiprocessing, os

current_directory = os.getcwd()

Lambda = 20
pid = 15
mass = 500

quirk_inputPathP = f"{current_directory}/4vector_{mass}GeV_PID{pid}_1jet.csv"
quirk_inputPathA = f"{current_directory}/4vector_{mass}GeV_PID{-pid}_1jet.csv"

quirk_path_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles")
os.makedirs(quirk_path_output, exist_ok=True)
quirk_path_output = os.path.join(quirk_path_output, f"QuirkMass_{mass}_Lambda_{Lambda}_multiprocessed.csv")


passed = quirk_multiprocessing.process(Lambda, quirk_inputPathP, quirk_inputPathA, quirk_path_output)

print(f"passed: {passed}")



background_path_input = f"{current_directory}/4vector_pionbgd_wCuts.csv"

background_path_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles", "Bgd")
os.makedirs(background_path_output, exist_ok=True)
background_path_output = os.path.join(background_path_output, "Bgd_500_1jet_wCuts_multiprocessed.csv")

background_multiprocessing.process(passed, background_path_input, background_path_output)