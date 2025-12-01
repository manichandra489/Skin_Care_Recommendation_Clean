import os
import tempfile

import streamlit as st
from PIL import Image
import numpy as np
import cv2

from imageContrast import analyze_face_contrast_bgr
from recommender import run, ImageState


# ---------- Page config & basic styling ----------
st.set_page_config(
    page_title="Skin Wise",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .small-image img {
        max-height: 280px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Skin Wise")
st.write("Analyze your face image for quality and get personalized L'Oréal skincare recommendations.")


# ---------- Source selector ----------
with st.sidebar:
    st.header("Input options")
    source = st.radio(
        "Choose image source",
        ("Upload image", "Use camera"),
        index=0,
    )

    st.markdown("---")
    st.caption("Tips for best results:\n- Face clearly visible\n- Good lighting\n- Look straight at the camera")


# ---------- Image acquisition ----------
uploaded_img = None

if source == "Upload image":
    file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if file is not None:
        uploaded_img = Image.open(file).convert("RGB")
else:
    camera_frame = st.camera_input("Take a photo")
    if camera_frame is not None:
        uploaded_img = Image.open(camera_frame).convert("RGB")


if uploaded_img is None:
    st.info("Please upload a face image or take a photo from the camera to begin.")
    st.stop()


# ---------- Layout: left = image + quality, right = recommendations ----------
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("Input image")
    st.container().markdown('<div class="small-image">', unsafe_allow_html=True)
    st.image(uploaded_img, caption="Your image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Convert to BGR for contrast checker
    rgb = np.array(uploaded_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    st.subheader("Image quality check")
    contrast_score, status = analyze_face_contrast_bgr(bgr)

    if status == "No face":
        st.error("No face detected. Please upload / capture a clear face photo.")
        recommend_enabled = False
    else:
        st.write(f"Face contrast score: **{contrast_score:.1f}**")
        st.write(f"Quality classification: **{status}**")
        if status == "Low":
            st.warning("Face contrast is low. Try better lighting or a sharper image for more accurate results.")
        recommend_enabled = (status == "High")


# ---------- Build ImageState (always as a temp path) ----------
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    temp_path = tmp.name
    uploaded_img.save(temp_path)

img_state = ImageState(image=temp_path)


# ---------- Recommendation section ----------
with right_col:
    st.subheader("Skin analysis & recommendations")

    if not recommend_enabled:
        st.info("Recommendations are enabled only when a clear, high‑contrast face is detected.")
    else:
        if st.button("Generate recommendations", type="primary", use_container_width=True):
            with st.spinner("Analyzing skin and finding suitable L'Oréal products..."):
                result = run(img_state)
                st.success("Recommendations generated")

                if isinstance(result, dict):
                    # If abnormal skin or no products/links, show doctor message
                    if result["medimage"] == []:
                        st.error(
                            "Your skin analysis shows signs that may require medical attention. "
                            "Please consult a dermatologist or healthcare professional for a proper diagnosis and treatment."
                        )
                    else:
                        if isinstance(result, dict):
                            if "skin_disease" in result and result["skin_disease"]:
                                st.markdown(f"**Skin condition:** {result['skin_disease']}")
                            if "bauman_type" in result:
                                st.markdown(f"**Bauman skin type:** {result['bauman_type']}")
                            if "des" in result:
                                st.markdown("### Recommended routine")
                                st.write(result["des"])
                            if "medimage" in result and result["medimage"]:
                                st.markdown("### Suggested products")
                                for url in result["medimage"]:
                                    st.markdown(f"- {url}")
                            if "toxic" in result and result["toxic"]:
                                st.markdown("### Ingredient compatibility")
                                st.write(result["toxic"])
                        else:
                            # Fallback: just print whatever the graph returned
                            st.write(result)
                                    # # Skin info
                        # if "skin_disease" in result and result["skin_disease"]:
                        #     st.markdown(f"**Skin condition:** {result['skin_disease']}")
                        # if "bauman_type" in result and result["bauman_type"]:
                        #     st.markdown(f"**Bauman skin type:** {result['bauman_type']}")
                        # # Source URLs used for scraping
                        # if "weblinks" in result and result["weblinks"]:
                        #     st.markdown("### Source pages")
                        #     for url in result["weblinks"]:
                        #         st.markdown(f"- {url}")

                        # # Top products with images
                        # if "products" in result and not result["products"].empty:
                        #     st.markdown("### Top recommended products")
                        #     df = result["products"]

                            # for _, row in df.iterrows():
                            #     st.markdown(f"**{row['Title']}**")
                            #     st.write(row["Subtitle"])
                            #     st.write(f"Price: {row['Price']}  ·  Rating: {row['Rating']}")

                            #     img = row.get("Img_url")
                            #     if isinstance(img, list):
                            #         img = img[0]
                            #     if img and img != "N/A":
                            #         st.image(img, width=200)

                            #     link = row.get("Link")
                            #     if link and link != "N/A":
                #             #         st.markdown(f"[View product]({link})")
                # else:
                #     st.write(result)



