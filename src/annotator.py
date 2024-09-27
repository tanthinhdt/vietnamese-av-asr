import polars as pl
import streamlit as st
from pathlib import Path


def add_columns(df) -> pl.DataFrame:
    if "female" not in df.columns:
        df = df.with_columns(female=pl.Series([False] * len(df)))
    if "dialect" not in df.columns:
        df = df.with_columns(dialect=pl.Series(["unknown"] * len(df)).cast(pl.String))
    if "english" not in df.columns:
        df = df.with_columns(english=pl.Series([False] * len(df)))
    if "done" not in df.columns:
        df = df.with_columns(done=pl.Series([False] * len(df)))
    if "error" not in df.columns:
        df = df.with_columns(error=pl.Series([False] * len(df)))
    return df


def reset_iterator() -> None:
    st.session_state.curr_idx = 0
    st.session_state.curr_row = st.session_state.df.row(
        st.session_state.curr_idx, named=True
    )
    st.session_state.to_id = st.session_state.curr_row["id"]
    st.session_state.to_idx = st.session_state.curr_idx


def load_metadata() -> pl.DataFrame:
    st.session_state.metadata_df = add_columns(st.session_state.metadata_df)

    st.session_state.shards = get_available_shards()
    st.session_state.splits = get_available_splits()
    st.session_state.available_df = (
        st.session_state.metadata_df
        .filter(
            (pl.col("shard").is_in(st.session_state.shards))
            & (pl.col("split").is_in(st.session_state.splits))
        )
        .sort(["channel", "id"])
    )
    st.session_state.df = st.session_state.available_df
    st.session_state.start_row = 0
    st.session_state.end_row = len(st.session_state.available_df)

    reset_iterator()


def select_subset() -> None:
    if st.session_state.subset == "100-hour":
        st.session_state.metadata_file_name = "metadata_100.parquet"
    elif st.session_state.subset == "200-hour":
        st.session_state.metadata_file_name = "metadata_200.parquet"
    else:
        st.session_state.metadata_file_name = "metadata.parquet"
    st.session_state.metadata_file = st.session_state.data_dir / st.session_state.metadata_file_name
    st.session_state.metadata_df = pl.read_parquet(st.session_state.metadata_file)
    load_metadata()


def upload_file() -> None:
    st.session_state.metadata_df = pl.read_parquet(st.session_state.uploaded_file.read())
    load_metadata()


def select_file_mode() -> None:
    if st.session_state.file_mode == "Default":
        if "subset" not in st.session_state:
            st.session_state.subset = "1000-hour"
        select_subset()


def get_available_shards() -> set:
    visual_dir = st.session_state.data_dir / "visual"
    audio_dir = st.session_state.data_dir / "audio"
    available_shards = (
        set([f.name for f in visual_dir.iterdir() if f.is_dir()])
        .intersection([f.name for f in audio_dir.iterdir() if f.is_dir()])
        .intersection(st.session_state.metadata_df["shard"].unique().to_list())
    )
    if len(available_shards) == 0:
        st.error("No shards found in the data directory.")
        st.stop()
        return set()
    return sorted(available_shards)


def get_available_splits() -> set:
    return sorted(st.session_state.metadata_df["split"].unique().to_list())


def filter() -> None:
    if len(st.session_state.shards) == 0:
        st.session_state.shards = get_available_shards()
    if len(st.session_state.splits) == 0:
        st.session_state.splits = get_available_splits()

    st.session_state.available_df = (
        st.session_state.metadata_df
        .filter(
            pl.col("shard").is_in(st.session_state.shards)
            & pl.col("split").is_in(st.session_state.splits)
        )
        .sort(["channel", "id"])
    )
    st.session_state.df = st.session_state.available_df

    st.session_state.start_row = 0
    st.session_state.end_row = len(st.session_state.available_df)

    reset_iterator()
    update_values()


def slice() -> None:
    st.session_state.df = st.session_state.available_df.slice(
        st.session_state.start_row,
        st.session_state.end_row - st.session_state.start_row,
    )
    reset_iterator()
    update_values()


def update_label(col_name: str) -> None:
    st.session_state.df[st.session_state.curr_idx, col_name] = st.session_state[col_name]


def update_values() -> None:
    col_names = [
        "female", "dialect", "english", "english", "error", "transcript", "done"
    ]
    for col_name in col_names:
        st.session_state[col_name] = st.session_state.curr_row[col_name]


def update_df() -> None:
    for idx, cols in st.session_state.data_editor["edited_rows"].items():
        for col_name, value in cols.items():
            st.session_state.df[idx, col_name] = value
    st.session_state.curr_row = st.session_state.df.row(
        st.session_state.curr_idx, named=True
    )
    update_values()


def iterate(direction: str, ids: list) -> None:
    if direction == "forward":
        st.session_state.curr_idx += 1
    elif direction == "backward":
        st.session_state.curr_idx -= 1
    st.session_state.curr_row = st.session_state.df.row(
        st.session_state.curr_idx, named=True
    )
    st.session_state.to_idx = st.session_state.curr_idx
    st.session_state.to_id = ids[st.session_state.curr_idx]
    update_values()


def to_id(ids: list) -> None:
    st.session_state.curr_idx = ids.index(st.session_state.to_id)
    st.session_state.curr_row = st.session_state.df.row(
        st.session_state.curr_idx, named=True
    )
    st.session_state.to_idx = st.session_state.curr_idx
    update_values()


def to_idx(ids: list) -> None:
    st.session_state.curr_idx = st.session_state.to_idx
    st.session_state.curr_row = st.session_state.df.row(
        st.session_state.curr_idx, named=True
    )
    st.session_state.to_id = ids[st.session_state.to_idx]
    update_values()


def merge_df() -> None:
    unchanged_columns = [
        "id", "shard", "split", "channel", "ori_video", "topic",
        "fps", "sampling_rate",
        "video_num_frames", "audio_num_frames",
    ]
    changed_columns = [
        "transcript", "female", "dialect", "english", "done", "error",
    ]
    st.session_state.metadata_df = (
        st.session_state.df
        .join(
            st.session_state.metadata_df,
            on=unchanged_columns,
            how="right",
        )
        .with_columns(
            [
                pl.col(col).fill_null(pl.col(f"{col}_right"))
                for col in changed_columns
            ]
        )
        .drop([f"{col}_right" for col in changed_columns])
    )


@st.dialog("Save as")
def save_as() -> None:
    @st.fragment
    def download(data) -> None:
        st.download_button(
            "Download",
            data=data,
            file_name="annotated_metadata.parquet",
            use_container_width=True,
            key="download_button",
            help="Save the annotations as a new file",
        )

    progress_bar = st.progress(0, "Merging dataframes...")
    merge_df()
    progress_bar.progress(1 / 3, "Converting to pandas...")
    pd_df = st.session_state.metadata_df.to_pandas()
    progress_bar.progress(2 / 3, "Exporting as parquet...")
    data = pd_df.to_parquet()
    progress_bar.progress(3 / 3, "Done")
    download(data)


def save() -> None:
    merge_df()
    st.session_state.metadata_df.write_parquet(st.session_state.metadata_file)


# Set the layout ===================================================================
# Set the page layout --------------------------------------------------------------
st.set_page_config(
    page_title="Video Annotator",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

label_area_col_1, label_area_col_2 = st.columns([3, 1])
label_area_height = 660

metadata_container = label_area_col_1.container(
    height=label_area_height,
    border=False,
)
data_display = metadata_container.container()
fields_input_cols = metadata_container.columns([3, 1, 1, 1])
dialect_input = fields_input_cols[0].container()
female_input = fields_input_cols[1].container()
english_input = fields_input_cols[2].container()
error_input = fields_input_cols[3].container()
metadata_col_1, metadata_col_2 = metadata_container.columns([2, 1])
transcript_input = metadata_col_1.container()
done_input = metadata_col_2.container()

media_container = label_area_col_2.container(
    height=label_area_height,
    border=False,
)
video_display = media_container.container()
audio_display = media_container.container()
media_area_col_1, media_area_col_2 = media_container.columns(2)
to_id_input = media_area_col_1.container()
to_idx_input = media_area_col_2.container()
next_id_button = media_container.container()
prev_id_button = media_container.container()

st.divider()

metric_cols = st.columns(8)
num_examples_display = metric_cols[0].empty()
progress_display = metric_cols[1].empty()
num_female_display = metric_cols[2].empty()
num_north_dialect_display = metric_cols[3].empty()
num_central_dialect_display = metric_cols[4].empty()
num_south_dialect_display = metric_cols[5].empty()
num_english_display = metric_cols[6].empty()
num_error_display = metric_cols[7].empty()

# Set the sidebar layout -----------------------------------------------------------
st.sidebar.header("Settings")
data_dir_input = st.sidebar.container()
file_mode_input = st.sidebar.container()
file_input = st.sidebar.container()
shard_input = st.sidebar.container()
split_input = st.sidebar.container()
num_examples_col_1, num_examples_col_2 = st.sidebar.columns(2)
start_row_input = num_examples_col_1.container()
end_row_input = num_examples_col_2.container()
start_annotating_button = st.sidebar.container()
stop_annotating_button = st.sidebar.container()
save_as_button = st.sidebar.container()
save_button = st.sidebar.container()


# Load metadata ====================================================================
# Input data directory -------------------------------------------------------------
data_dir = data_dir_input.text_input(
    "Data Directory",
    value="data/raw/vasr",
    help="Directory containing the videos to be annotated",
)
st.session_state.data_dir = Path(data_dir)
visual_dir = st.session_state.data_dir / "visual"
if not visual_dir.exists():
    st.error("Visual directory not found.")
    st.stop()
audio_dir = st.session_state.data_dir / "audio"
if not audio_dir.exists():
    st.error("Audio directory not found.")
    st.stop()
# Input file mode ------------------------------------------------------------------
file_mode_input.selectbox(
    label="File mode",
    options=["Default", "Uploading"],
    index=0,
    on_change=select_file_mode,
    key="file_mode",
    help="Select the file mode",
)
# Input subset ---------------------------------------------------------------------
if st.session_state.file_mode == "Default":
    file_input.selectbox(
        label="Subset",
        options=["1000-hour", "100-hour", "200-hour"],
        on_change=select_subset,
        key="subset",
        help="Select the subset to annotate",
    )
# Upload metadata ------------------------------------------------------------------
elif st.session_state.file_mode == "Uploading":
    file_input.file_uploader(
        label="Upload metadata",
        type=["parquet"],
        on_change=upload_file,
        key="uploaded_file",
        help="Upload a metadata file",
    )
# Load metadata --------------------------------------------------------------------
if "metadata_file_name" not in st.session_state:
    if st.session_state.subset == "100-hour":
        st.session_state.metadata_file_name = "metadata_100.parquet"
    elif st.session_state.subset == "200-hour":
        st.session_state.metadata_file_name = "metadata_200.parquet"
    else:
        st.session_state.metadata_file_name = "metadata.parquet"
if "metadata_file" not in st.session_state:
    st.session_state.metadata_file = st.session_state.data_dir / st.session_state.metadata_file_name
    if not st.session_state.metadata_file.exists():
        st.error("Metadata file not found.")
        st.stop()

if "metadata_df" not in st.session_state:
    st.session_state.metadata_df = pl.read_parquet(st.session_state.metadata_file)
    st.session_state.metadata_df = add_columns(st.session_state.metadata_df)

if "shards" not in st.session_state:
    st.session_state.shards = get_available_shards()
if "splits" not in st.session_state:
    st.session_state.splits = get_available_splits()
if "available_df" not in st.session_state:
    st.session_state.available_df = (
        st.session_state.metadata_df
        .filter(
            (pl.col("shard").is_in(st.session_state.shards))
            & (pl.col("split").is_in(st.session_state.splits))
        )
        .sort(["channel", "id"])
    )
if "start_row" not in st.session_state:
    st.session_state.start_row = 0
if "end_row" not in st.session_state:
    st.session_state.end_row = len(st.session_state.available_df)
if "df" not in st.session_state:
    st.session_state.df = st.session_state.available_df

# Filter metadata ==================================================================
shard_input.multiselect(
    "Shard(s)",
    options=get_available_shards(),
    default=st.session_state.shards,
    on_change=filter,
    key="shards",
    help="Select the shard to annotate",
)
split_input.multiselect(
    "Split",
    options=get_available_splits(),
    default=st.session_state.splits,
    on_change=filter,
    key="splits",
    help="Select the split to annotate",
)
start_row_input.number_input(
    "Start row",
    min_value=0,
    max_value=st.session_state.end_row,
    value=st.session_state.start_row,
    step=100,
    on_change=slice,
    key="start_row",
    help="Start row of the table",
)
end_row_input.number_input(
    "End row",
    min_value=st.session_state.start_row + 1,
    max_value=len(st.session_state.available_df),
    value=st.session_state.end_row,
    step=100,
    on_change=slice,
    key="end_row",
    help="End row of the table",
)

# Get current row ==================================================================
ids = st.session_state.df["id"].to_list()
if "curr_idx" not in st.session_state:
    st.session_state.curr_idx = 0
if "curr_row" not in st.session_state:
    st.session_state.curr_row = st.session_state.df.row(
        st.session_state.curr_idx, named=True
    )
if "to_id" not in st.session_state:
    st.session_state.to_id = ids[st.session_state.curr_idx]
if "to_idx" not in st.session_state:
    st.session_state.to_idx = st.session_state.curr_idx

# Show the metadata input fields ===================================================
female_input.checkbox(
    label="Is female",
    value=st.session_state.curr_row["female"],
    on_change=update_label,
    kwargs={"col_name": "female"},
    key="female",
    help="Check if the speaker in the video is female.",
)
dialect_input.radio(
    label="Dialect",
    options=["northern", "central", "southern", "unknown"],
    horizontal=True,
    index=["northern", "central", "southern", "unknown"].index(st.session_state.curr_row["dialect"]),
    on_change=update_label,
    kwargs={"col_name": "dialect"},
    key="dialect",
    help="Select the dialect of the speaker in the video.",
)
english_input.checkbox(
    label="Has English",
    value=st.session_state.curr_row["english"],
    on_change=update_label,
    kwargs={"col_name": "english"},
    key="english",
    help="Check if the transcript has English words.",
)
error_input.checkbox(
    label="Has error",
    value=st.session_state.curr_row["error"],
    on_change=update_label,
    kwargs={"col_name": "error"},
    key="error",
    help="Check if the example has an error.",
)
transcript_input.text_area(
    label="Transcript",
    value=st.session_state.curr_row["transcript"],
    on_change=update_label,
    kwargs={"col_name": "transcript"},
    key="transcript",
    help="Transcript of the video.",
)
done_input.checkbox(
    label="Is done",
    value=st.session_state.curr_row["done"],
    on_change=update_label,
    kwargs={"col_name": "done"},
    key="done",
    help="Check if the video has been annotated.",
)

# Show the metadata view ===========================================================
data_display.data_editor(
    st.session_state.df,
    use_container_width=True,
    hide_index=False,
    column_order=[
        "id", "transcript", "female", "dialect", "english", "error", "done", "channel",
    ],
    on_change=update_df,
    key="data_editor",
    column_config={
        "id": st.column_config.Column(
            label="ID",
            width="small",
            disabled=True,
            help="Unique identifier for the video",
        ),
        "shard": None,
        "split": None,
        "channel": st.column_config.TextColumn(
            label="Channel",
            width="small",
            disabled=True,
            help="Channel number of the video",
        ),
        "ori_video": None,
        "topic": None,
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
            options=["northern", "central", "southern", "unknown"],
            help="Select the dialect of the speaker in the video",
        ),
        "english": st.column_config.CheckboxColumn(
            label="English",
            width="small",
            help="Check if the transcript has English words",
        ),
        "done": st.column_config.CheckboxColumn(
            label="Done",
            width="small",
            help="Check if the video has been annotated",
        ),
        "error": st.column_config.CheckboxColumn(
            label="Error",
            width="small",
            help="Check if the example has an error",
        ),
    },
)

# Show the video and audio =========================================================
visual_path = (
    st.session_state.data_dir / "visual"
    / st.session_state.curr_row["shard"]
    / f"{st.session_state.curr_row['id']}.mp4"
)
if visual_path.exists():
    video_display.video(str(visual_path), autoplay=True, loop=True)
else:
    video_display.error(f"Video file {visual_path} does not exist.")

audio_path = (
    st.session_state.data_dir / "audio"
    / st.session_state.curr_row["shard"]
    / f"{st.session_state.curr_row['id']}.wav"
)
if audio_path.exists():
    audio_display.audio(str(audio_path), autoplay=True, loop=True)
else:
    audio_display.error(f"Audio file {audio_path} does not exist.")

# Show navigation buttons ==========================================================
lock_next_id = any(
    [
        not st.session_state.df[st.session_state.curr_idx, "done"],
        st.session_state.curr_idx == len(st.session_state.df) - 1,
    ]
)
next_id_button.button(
    "Next example",
    use_container_width=True,
    on_click=iterate,
    kwargs={"direction": "forward", "ids": ids},
    disabled=lock_next_id,
    help="Go to the next example.",
)

prev_id_button.button(
    "Previous example",
    use_container_width=True,
    on_click=iterate,
    kwargs={"direction": "backward", "ids": ids},
    disabled=st.session_state.curr_idx == 0,
    help="Go to the previous example.",
)

to_id_input.selectbox(
    "To ID",
    options=ids,
    on_change=to_id,
    kwargs={"ids": ids},
    key="to_id",
    help="Go to a specific ID.",
)

to_idx_input.number_input(
    "To index",
    min_value=0,
    max_value=len(ids) - 1,
    value=st.session_state.curr_idx,
    on_change=to_idx,
    kwargs={"ids": ids},
    key="to_idx",
    help="Go to a specific index.",
)

# Show the metrics =================================================================
num_examples = len(st.session_state.df)
num_examples_display.metric(
    "Examples",
    value=f"{num_examples:,}",
)
progress_display.metric(
    "Progress",
    value=f"{len(st.session_state.df.filter(pl.col('done'))) / num_examples:.2f}%"
)
num_female_display.metric(
    "Female examples",
    value=f"{len(st.session_state.df.filter(pl.col('female'))):,}",
)
num_north_dialect_display.metric(
    "Northern dialect",
    value=f"{len(st.session_state.df.filter(pl.col('dialect') == 'northern')):,}",
)
num_central_dialect_display.metric(
    "Central dialect",
    value=f"{len(st.session_state.df.filter(pl.col('dialect') == 'central')):,}",
)
num_south_dialect_display.metric(
    "Southern dialect",
    value=f"{len(st.session_state.df.filter(pl.col('dialect') == 'southern')):,}",
)
num_english_display.metric(
    "English examples",
    value=f"{len(st.session_state.df.filter(pl.col('english'))):,}",
)
num_error_display.metric(
    "Error examples",
    value=f"{len(st.session_state.df.filter(pl.col('error'))):,}",
)

# Save the annotations ============================================================
save_as_button.button(
    "Save as",
    on_click=save_as,
    use_container_width=True,
    help="Save the annotations as a new file",
)
save_button.button(
    "Save",
    on_click=save,
    use_container_width=True,
    disabled=st.session_state.file_mode == "Uploading",
    help="Save the annotations",
)
