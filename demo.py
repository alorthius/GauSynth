import gradio as gr

from demo_fns import *


with gr.Blocks() as demo:
    with gr.Row():

        # Block 1
        with gr.Column():
            gr.Markdown("### Directory and Video Player")

            dir_name = gr.Textbox(label="Directory")
            dir_butt = gr.Button(value="Create directory")

            vid_input = gr.Video(label="Select a video")
            with gr.Row():
                fps = gr.Textbox(label="Original fps")
                process_vid_butt = gr.Button(value="Process video")

            orig_video = gr.Video(label="Preprocessed video", interactive=False)

            process_colmap_butt = gr.Button(value="Run SfM")
            # TODO: point-cloud visualization

        # Block 2
        with gr.Column():
            gr.Markdown("### Image Generator and Text Output")

            num_frames_sheet = gr.Radio(choices=[2, 3, 4], label="Select character sheet dim")
            create_sheet_butt = gr.Button(value="Create character sheet")
            orig_sheet = gr.Image(label="Original character sheet", interactive=False)

            prompt = gr.Textbox(label="Text prompt")
            # TODO: add fooocus params
            reimagine_butt = gr.Button(value="Reimagine")
            reimagine_sheet = gr.Image(label="Reimagined character sheet", interactive=False)

        # Block 3
        with gr.Column():
            gr.Markdown("### Video Player")

            ebsynth_butt = gr.Button(value="Interpolate frames")
            reimagined_vid = gr.Video(label="Reimagined frames", interactive=False)

            sr_butt = gr.Button(value="Super Resolution")
            sr_vid = gr.Video(label="SR frames", interactive=False)

        # Block 4
        with gr.Column():
            gr.Markdown("### Video Player")

            gs_butt = gr.Button(value="Run 3D GS reconstruction")
            gs_renders = gr.Video(label="Rendered frames", interactive=False)

    dir_butt.click(
        fn=create_dir,
        inputs=dir_name,
        outputs=None,
    )

    process_vid_butt.click(
        fn=process_video,
        inputs=[fps, vid_input, dir_name],
        outputs=orig_video,
    )

    create_sheet_butt.click(
        fn=create_sheet,
        inputs=[num_frames_sheet, dir_name],
        outputs=orig_sheet,
    )

    reimagine_butt.click(
        fn=reimagine,
        inputs=[orig_sheet, prompt],
        outputs=reimagine_sheet,
    )

demo.launch()
