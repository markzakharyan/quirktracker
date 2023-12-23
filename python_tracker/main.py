import argparse
import quirk_multiprocessing, background_multiprocessing, os

def main(quirk_inputPathP, quirk_inputPathA, quirk_path_output, background_path_input, background_path_output, Lambda, mass, pid):


    passed = quirk_multiprocessing.process(Lambda, quirk_inputPathP, quirk_inputPathA, quirk_path_output)

    print(f"passed: {passed}")


    background_multiprocessing.process(passed, background_path_input, background_path_output)


if __name__ == '__main__':

    Lambda = 20
    pid = 15
    mass = 500


    quirk_inputPathP = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"4vector_{mass}GeV_PID{pid}_1jet.csv")
    quirk_inputPathA = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"4vector_{mass}GeV_PID{-pid}_1jet.csv")
    quirk_path_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles", f"QuirkMass_{mass}_Lambda_{Lambda}_multiprocessed.csv")

    background_path_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4vector_pionbgd_wCuts.csv")
    background_path_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HitFiles", "Bgd", "Bgd_500_1jet_wCuts_multiprocessed.csv")



    parser = argparse.ArgumentParser(description='Run simulation with quirk and background processes.')

    parser.add_argument('-qp', '--quirk_inputPathP', type=str, help='The path of the first Quirk CSV file.', metavar='PATH')
    parser.add_argument('-qa', '--quirk_inputPathA', type=str, help='The path of the second Quirk CSV file.', metavar='PATH')
    parser.add_argument('-qo', '--quirk_path_output', type=str, help='Output Quirk CSV file path.', metavar='PATH')

    parser.add_argument('-bi', '--background_path_input', type=str, help='Input background CSV path.', metavar='PATH')
    parser.add_argument('-bo', '--background_path_output', type=str, help='Output background CSV file path.', metavar='PATH')

    parser.add_argument('-L', '--Lambda', type=float, help='Lambda used Quirk string tension (root_sigma).', metavar='FLOAT', default=Lambda)
    parser.add_argument('-m', '--mass', type=float, help='Mass used for quirks (GeV).', metavar='FLOAT', default=mass)
    parser.add_argument('-p', '--pid', type=int, help='Positive PID for quirks (should be 15, weird if you change this)', metavar='INT', default=pid)

    args = parser.parse_args()


    # Here because I want things like mass, pid, and lambda to update the default values of the other unspecified variables
    arg_vars = {
        'quirk_inputPathP': None,
        'quirk_inputPathA': None,
        'quirk_path_output': None,
        'background_path_input': None,
        'background_path_output': None
    }
    for i in arg_vars:
        if getattr(args, i) is not None:
            globals()[i] = getattr(args, i)

    

    Lambda = args.Lambda
    pid = args.pid
    mass = args.mass    


    main(quirk_inputPathP, quirk_inputPathA, quirk_path_output, background_path_input, background_path_output, Lambda, mass, pid)