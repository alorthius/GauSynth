from Ezsynth.ezsynth import Ezsynth


def interpolate(key_path, input_folder, output_folder):
    styles = [key_path]

    ebsynth_model = Ezsynth(
        styles=styles,
        imgsequence=input_folder,
        edge_method="PAGE",
        flow_method="RAFT",
        model="sintel",
        output_folder=output_folder,
    )
    ebsynth_model.run()

    results = ebsynth_model.results
    return results
