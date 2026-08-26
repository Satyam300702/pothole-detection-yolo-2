import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Pothole Detection & Tracking")
st.write("Upload a road video to detect, track and count potholes.")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = YOLO("best (16).pt")

    return model


model = load_model()

# ==========================================
# UPLOAD VIDEO
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a road video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ==========================================
# SETTINGS
# ==========================================

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.70,
    step=0.05
)

image_size = st.selectbox(
    "Image Size",
    [640, 768, 960],
    index=2
)

# ==========================================
# PROCESS VIDEO
# ==========================================

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("🚀 Detect & Count Potholes"):

        # --------------------------------------
        # SAVE UPLOADED VIDEO TEMPORARILY
        # --------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_file.read()
        )

        input_file.close()

        input_path = input_file.name

        # --------------------------------------
        # OUTPUT FILE
        # --------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_file.close()

        output_path = output_file.name

        # --------------------------------------
        # VIDEO PROCESSING
        # --------------------------------------

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            st.error("Could not open the uploaded video.")

        else:

            fps = cap.get(cv2.CAP_PROP_FPS)

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            # ----------------------------------
            # VIDEO WRITER
            # ----------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # ----------------------------------
            # PROGRESS
            # ----------------------------------

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            progress_bar = st.progress(0)

            status = st.empty()

            frame_number = 0

            # ==================================
            # POTHOLE COUNTING
            # ==================================

            # Stores every unique tracking ID
            unique_pothole_ids = set()

            # ==================================
            # PROCESS FRAMES
            # ==================================

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # ----------------------------------
                # YOLO TRACKING
                # ----------------------------------

                results = model.track(
                    frame,
                    conf=confidence,
                    imgsz=image_size,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False
                )

                # ----------------------------------
                # GET TRACKING IDs
                # ----------------------------------

                current_track_ids = []

                if results[0].boxes.id is not None:

                    track_ids = (
                        results[0]
                        .boxes
                        .id
                        .int()
                        .cpu()
                        .tolist()
                    )

                    current_track_ids = track_ids

                    # Add IDs to unique set
                    for track_id in track_ids:

                        unique_pothole_ids.add(
                            track_id
                        )

                # ----------------------------------
                # COUNTS
                # ----------------------------------

                total_potholes = len(
                    unique_pothole_ids
                )

                current_potholes = len(
                    current_track_ids
                )

                # ----------------------------------
                # DRAW YOLO RESULTS
                # ----------------------------------

                annotated_frame = results[0].plot()

                # ----------------------------------
                # DISPLAY TOTAL COUNT ON VIDEO
                # ----------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (10, 10),
                    (330, 65),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    annotated_frame,
                    f"Total Potholes: {total_potholes}",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # ----------------------------------
                # WRITE FRAME
                # ----------------------------------

                writer.write(
                    annotated_frame
                )

                # ----------------------------------
                # PROGRESS
                # ----------------------------------

                frame_number += 1

                if total_frames > 0:

                    progress = (
                        frame_number /
                        total_frames
                    )

                    progress_bar.progress(
                        progress
                    )

                    status.write(
                        f"Processing frame "
                        f"{frame_number}/{total_frames}"
                        f" | Unique potholes: "
                        f"{total_potholes}"
                    )

            # ==================================
            # RELEASE
            # ==================================

            cap.release()
            writer.release()

            progress_bar.progress(1.0)

            status.success(
                "Pothole detection and counting completed!"
            )

            # ==================================
            # FINAL COUNT
            # ==================================

            st.subheader("📊 Pothole Statistics")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Total Unique Potholes",
                    len(unique_pothole_ids)
                )

            with col2:

                st.metric(
                    "Tracking IDs",
                    len(unique_pothole_ids)
                )

            # ==================================
            # DISPLAY RESULT
            # ==================================

            st.subheader(
                "🎥 Processed Video"
            )

            with open(
                output_path,
                "rb"
            ) as video_file:

                video_bytes = video_file.read()

            st.video(video_bytes)

            # ==================================
            # DOWNLOAD
            # ==================================

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="pothole_detection.mp4",
                mime="video/mp4"
            )

        # ======================================
        # CLEANUP
        # ======================================

        if os.path.exists(input_path):

            os.remove(input_path)

        if os.path.exists(output_path):

            os.remove(output_path)