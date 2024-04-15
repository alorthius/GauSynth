import gradio as gr

from demo_fns import *


with gr.Blocks() as demo:
    gr.Markdown("### Interpolation & Super Resolution")
    with gr.Row():

        # Block 1
        with gr.Column():
            gr.Markdown("### Video Preprocessing & SfM")

            dir_name = gr.Textbox(label="Directory")
            dir_butt = gr.Button(value="Create directory")

            vid_input = gr.Video(label="Select a video")
            with gr.Row():
                fps = gr.Textbox(label="Original fps")
                process_vid_butt = gr.Button(value="Process video")
            orig_video = gr.Video(label="Preprocessed video", interactive=False)

            process_colmap_butt = gr.Button(value="Run SfM")
            colmap_video = gr.Video(label="Sparse Colmap reconstruction", interactive=False)

            colmap_table = gr.DataFrame(
                [["-" for i in range(3)]],
                headers=["Cameras", "Images", "Points"],
                label="Colmap reconstruction info",
                interactive=False,
            )

        # Block 2
        with gr.Column():
            gr.Markdown("### Keyframes Reimagination")

            num_frames_sheet = gr.Radio(choices=[2, 3, 4], label="Select character sheet dim")
            create_sheet_butt = gr.Button(value="Create character sheet")
            orig_sheet = gr.Image(label="Original character sheet", interactive=False)
            orig_sheet_file = gr.Textbox(label="Sheet filename", visible=False)  # temp for saving sheet filename

            prompt = gr.Textbox(label="Text prompt")
            # TODO: add fooocus params
            reimagine_butt = gr.Button(value="Reimagine")
            reimagine_sheet = gr.Image(label="Reimagined character sheet", interactive=False)
            reimagine_sheet_file = gr.Textbox(label="Sheet filename", visible=False)  # temp for saving sheet filename

        # Block 3
        with gr.Column():
            gr.Markdown("### Interpolation & Super Resolution")

            ebsynth_butt = gr.Button(value="Interpolate frames")
            reimagined_vid = gr.Video(label="Interpolated frames", interactive=False)

            post_proc_butt = gr.Button(value="Remove background")
            post_proc_vid = gr.Video(label="Post processed frames", interactive=False)

            sr_butt = gr.Button(value="Super Resolution")
            sr_vid = gr.Video(label="SR frames", interactive=False)

        # Block 4
        with gr.Column():
            gr.Markdown("### 3D Gaussian Splatting reconstruction")

            gs_iters = gr.Slider(label="Training iterations", minimum=5000, maximum=15000, step=500, value=10000)
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

    process_colmap_butt.click(
        fn=run_sfm,
        inputs=dir_name,
        outputs=[colmap_video, colmap_table],
    )

    create_sheet_butt.click(
        fn=create_sheet,
        inputs=[num_frames_sheet, dir_name],
        outputs=[orig_sheet, orig_sheet_file]
    )

    reimagine_butt.click(
        fn=reimagine,
        inputs=[orig_sheet, dir_name, orig_sheet_file, prompt],
        outputs=[reimagine_sheet, reimagine_sheet_file]
    )

    ebsynth_butt.click(
        fn=interpolate_frames,
        inputs=[reimagine_sheet_file, dir_name, num_frames_sheet],
        outputs=reimagined_vid,
    )

    post_proc_butt.click(
        fn=ebsynth_post_process,
        inputs=[dir_name],
        outputs=post_proc_vid,
    )

    sr_butt.click(
        fn=run_sr,
        inputs=[dir_name],
        outputs=sr_vid,
    )

    gs_butt.click(
        fn=gs_reconstruct,
        inputs=[dir_name, gs_iters],
        outputs=gs_renders,
    )

demo.launch()
