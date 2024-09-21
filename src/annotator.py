import polars as pl
import streamlit as st
from pathlib import Path


def metadata_view():
    with st.session_state.metadata_col:
        container = st.container(height=st.session_state.container_height, border=False)
        st.session_state.df = container.data_editor(
            st.session_state.df,
            height=st.session_state.container_height - 10,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            column_order=["id", "transcript", "female", "dialect", "done"],
            column_config={
                "id": st.column_config.Column(
                    label="ID",
                    width="small",
                    disabled=True,
                    help="Unique identifier for the video",
                ),
                "shard": None,
                "split": None,
                "fps": None,
                "sampling_rate": None,
                "video_num_frames": None,
                "audio_num_frames": None,
                "transcript": st.column_config.TextColumn(
                    label="Transcript",
                    width="large",
                    required=True,
                    help="Transcript of the video",
                ),
                "female": st.column_config.CheckboxColumn(
                    label="Female",
                    width="small",
                    help="Check if the speaker in the video is female",
                ),
                "dialect": st.column_config.SelectboxColumn(
                    label="Dialect",
                    width="small",
                    options=["northern", "central", "southern"],
                    help="Select the dialect of the speaker in the video",
                ),
                "done": st.column_config.CheckboxColumn(
                    label="Done",
                    width="small",
                    help="Check if the video has been annotated",
                ),
            },
        )


def media_view():
    def iter_forward():
        st.session_state.curr_idx += 1

    def iter_backward():
        st.session_state.curr_idx -= 1

    def to_id():
        idx = st.session_state.df["id"].to_list().index(st.session_state.to_id)
        st.session_state.update(curr_idx=idx)

    def to_idx():
        st.session_state.update(curr_idx=st.session_state.to_idx)

    with st.session_state.media_col:
        container = st.container(height=st.session_state.container_height, border=False)

        ids = st.session_state.df["id"].to_list()
        if "curr_idx" not in st.session_state:
            st.session_state.curr_idx = 0
        row = st.session_state.df.row(st.session_state.curr_idx, named=True)

        visual_path = st.session_state.data_dir / "visual" / row["shard"] / f"{row['id']}.mp4"
        audio_path = st.session_state.data_dir / "audio" / row["shard"] / f"{row['id']}.wav"
        if visual_path.exists():
            container.video(str(visual_path), autoplay=True, loop=True)
            container.audio(str(audio_path), autoplay=True, loop=True)
        else:
            st.error(f"Video file {visual_path} does not exist.")

        lock_next_id = any(
            [
                not row["done"],
                st.session_state.curr_idx == len(st.session_state.df) - 1,
            ]
        )
        container.button(
            "Next ID",
            on_click=iter_forward,
            use_container_width=True,
            disabled=lock_next_id,
            help="Go to the next ID.",
        )
        container.button(
            "Previous ID",
            on_click=iter_backward,
            use_container_width=True,
            disabled=st.session_state.curr_idx == 0,
            help="Go to the previous ID.",
        )

        col_1, col_2 = container.columns(2)
        col_1.selectbox(
            "To ID",
            options=ids,
            on_change=to_id,
            key="to_id",
            help="Go to a specific ID.",
        )
        col_2.number_input(
            "To index",
            min_value=0,
            max_value=len(ids) - 1,
            value=st.session_state.curr_idx,
            on_change=to_idx,
            key="to_idx",
            help="Go to a specific index.",
        )


def app():
    st.set_page_config(
        page_title="Video Annotator",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    metadata_col, media_col = st.columns([3, 1])
    st.session_state.metadata_col = metadata_col
    st.session_state.media_col = media_col
    st.session_state.container_height = 640

    st.divider()

    col_1, col_2, col_3, col_4, col_5, col_6, col_7 = st.columns(7)
    st.session_state.num_examples_display = col_1.empty()
    st.session_state.progress_display = col_2.empty()
    st.session_state.num_male_display = col_3.empty()
    st.session_state.num_female_display = col_4.empty()
    st.session_state.num_north_dialect_display = col_5.empty()
    st.session_state.num_central_dialect_display = col_6.empty()
    st.session_state.num_south_dialect_display = col_7.empty()

    st.sidebar.header("Settings")

    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="data/raw/vasr",
        help="Directory containing the videos to be annotated",
    )
    st.session_state.data_dir = Path(data_dir)

    uploaded_file = st.sidebar.file_uploader(
        "Upload annotation file",
        type=["parquet"],
        help="Upload a Parquet file containing annotations",
    )
    if uploaded_file is not None:
        st.session_state.metadata_df = pl.read_parquet(uploaded_file.read())
    elif "metadata_df" not in st.session_state:
        st.session_state.metadata_df = pl.read_parquet(
            st.session_state.data_dir / "metadata.parquet"
        )

    visual_dir = st.session_state.data_dir / "visual"
    audio_dir = st.session_state.data_dir / "audio"
    available_shards = (
        set([f.name for f in visual_dir.iterdir() if f.is_dir()])
        .intersection([f.name for f in audio_dir.iterdir() if f.is_dir()])
        .intersection(st.session_state.metadata_df["shard"].to_list())
    )
    shard_id = st.sidebar.selectbox(
        "Shard ID",
        options=list(available_shards) + ["all"],
        help="Select the shard to annotate",
    )
    if shard_id != "all":
        st.session_state.df = st.session_state.metadata_df.filter(
            pl.col("shard") == shard_id
        )
    else:
        st.session_state.df = st.session_state.metadata_df

    split = st.sidebar.selectbox(
        "Split",
        options=["all", "train", "valid", "test"],
        help="Select the split to annotate",
    )
    if split != "all":
        st.session_state.df = st.session_state.df.filter(pl.col("split") == split)

    num_rows = st.sidebar.number_input(
        "Number of rows",
        min_value=1,
        max_value=len(st.session_state.df),
        value=len(st.session_state.df),
        step=100,
        help="Number of rows to display in the metadata table",
    )
    st.session_state.df = st.session_state.df.head(num_rows)

    st.session_state.export_button = st.sidebar.container()

    if "done" not in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.with_columns(
            done=pl.Series([False] * len(st.session_state.df))
        )
    if "female" not in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.with_columns(
            female=pl.Series([False] * len(st.session_state.df))
        )
    if "dialect" not in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.with_columns(
            dialect=pl.Series([None] * len(st.session_state.df))
        )
    st.session_state.df = st.session_state.df.sort("id")

    metadata_view()
    media_view()

    num_examples = len(st.session_state.df)
    st.session_state.num_examples_display.metric(
        "Examples",
        value=f"{num_examples:,}",
    )
    st.session_state.progress_display.metric(
        "Progress",
        value=f"{len(st.session_state.df.filter(pl.col('done'))) / num_examples:.2f}%"
    )
    st.session_state.num_male_display.metric(
        "Male examples",
        value=f"{len(st.session_state.df.filter(~pl.col('female'))):,}",
    )
    st.session_state.num_female_display.metric(
        "Female examples",
        value=f"{len(st.session_state.df.filter(pl.col('female'))):,}",
    )
    st.session_state.num_north_dialect_display.metric(
        "Northern dialect",
        value=f"{len(st.session_state.df.filter(pl.col('dialect') == 'north')):,}",
    )
    st.session_state.num_central_dialect_display.metric(
        "Central dialect",
        value=f"{len(st.session_state.df.filter(pl.col('dialect') == 'central')):,}",
    )
    st.session_state.num_south_dialect_display.metric(
        "Southern dialect",
        value=f"{len(st.session_state.df.filter(pl.col('dialect') == 'south')):,}",
    )

    st.session_state.export_button.download_button(
        "Export annotations",
        data=st.session_state.df.drop("done").to_pandas().to_parquet(),
        file_name="annotated_metadata.parquet",
        use_container_width=True,
        help="Export the annotations to the output annotation file",
    )


if __name__ == "__main__":
    app()
