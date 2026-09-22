# ============================================================
# EDUCATIONAL MODULE OVERVIEW & CONCEPT EXPLANATIONS
# ============================================================
#
# WHAT IS STREAMLIT (streamlit)?
# Streamlit is an open-source Python library used to convert data science scripts 
# and machine learning workflows into interactive graphical web user interfaces (UIs) 
# without requiring web development languages like HTML, CSS, or JavaScript.
# - Why we use it: It acts as the "front-end" or visual presentation layer of our 
#   AI application, allowing non-technical users to input data via graphical sliders 
#   and text fields and view live predictions immediately.
#
# WHAT IS PANDAS (pandas)?
# Pandas is a powerful Python library used for data manipulation, tabular data 
# management, and data structuring.
# - Why we use it: Machine learning models expect input data formatted in exact 
#   structured grid structures (rows and columns). Pandas allows us to take the individual 
#   user inputs from the web UI and assemble them into a structured "DataFrame" 
#   that perfectly matches the exact tabular structure used during model training.
#
# WHAT IS JOBLIB (joblib)?
# Joblib is a set of tools in Python optimized for saving (serializing) and 
# loading (deserializing) Python objects—especially large numpy arrays and 
# trained scikit-learn Machine Learning models—to and from disk files (.pkl files).
# - Why we use it: Training a machine learning model takes time and computing power. 
#   Instead of re-training the model every time a user opens the web app, we save 
#   the trained model once to a file using Joblib and reload it instantly in memory.
#
# WHAT IS OS (os)?
# OS is a built-in Python standard library module that provides operating-system-level 
# functions, such as verifying file directories, checking file existence, and managing 
# system file paths across different operating systems (Windows, Linux, macOS).
# - Why we use it: It allows our code to safely check whether the saved trained model file 
#   actually exists on the user's hard drive before attempting to load it.
# ============================================================

# Line 1: Import the Streamlit web app library and assign it the alias 'st' for short syntax.
import streamlit as st

# Line 2: Import the Pandas data manipulation library and assign it the alias 'pd'.
import pandas as pd

# Line 3: Import joblib to read/write serialized trained ML model files from disk.
import joblib

# Line 4: Import os to interact with operating system directory structures and file paths.
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# Streamlit's page configuration command establishes the core web browser tab properties.
# It MUST be the very first Streamlit command invoked in the script execution flow.

st.set_page_config(
    # Set the textual title displayed on the browser tab tab-bar
    page_title="Student Stress Detection System",
    
    # Set the icon/emoji displayed next to the browser tab title
    page_icon="🧠",
    
    # Choose "wide" layout mode to stretch page elements across the full screen width
    layout="wide",
    
    # Ensure the sidebar control panel opens in an expanded state when the page loads
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS (Cascading Style Sheets)
# ============================================================
# HTML and CSS are standard languages for web layout and design styling.
# We inject custom CSS into Streamlit to override default margins, set custom fonts, 
# rounded borders, padding, and text colors for custom UI boxes.

st.markdown(
    """
    <style>

    /* Class styling for the main title at the top of the web page */
    .main-title {
        font-size: 38px;          /* Sets text size in pixels */
        font-weight: 700;          /* Makes text bold (700 font weight) */
        text-align: center;        /* Aligns text to the horizontally middle */
        margin-bottom: 5px;        /* Spacing beneath the element */
    }

    /* Class styling for the subtitle description below the main title */
    .subtitle {
        text-align: center;        /* Center-aligns subtitle text */
        font-size: 18px;          /* Sets readable secondary font size */
        color: #666666;            /* Sets text color to a soft grey hue */
        margin-bottom: 30px;       /* Adds generous vertical spacing beneath subtitle */
    }

    /* Custom styled card container for displaying the model prediction output */
    .result-box {
        padding: 25px;             /* Adds internal breathing room around text */
        border-radius: 12px;       /* Creates soft rounded corner borders */
        text-align: center;        /* Centers output text horizontally */
        margin-top: 20px;          /* Spacing above the output box */
        border: 1px solid #dddddd; /* Adds a thin light-grey border around the box */
    }

    /* Header styling within the prediction result box */
    .result-title {
        font-size: 20px;          /* Medium prominent heading font size */
        font-weight: 600;          /* Semi-bold text styling */
        margin-bottom: 10px;       /* Vertical gap below title */
    }

    /* Display styling for the predicted classification text label */
    .result-value {
        font-size: 32px;          /* Large prominent emphasis font size */
        font-weight: 700;          /* Heavy bold emphasis */
    }

    /* Styling for general informative background callout boxes */
    .info-box {
        padding: 15px;             /* Internal container spacing */
        border-radius: 10px;       /* Rounded corners */
        background-color: #f5f5f5; /* Light grey background fill */
        margin-top: 10px;          /* Top exterior spacing */
        margin-bottom: 10px;       /* Bottom exterior spacing */
    }

    </style>
    """,
    # unsafe_allow_html=True allows raw HTML/CSS strings to be rendered directly into the DOM
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATH
# ============================================================
# Absolute system file path referencing where the saved trained model file (.pkl) is stored.
# The raw-string prefix 'r' tells Python to treat backslashes '\' as raw characters 
# rather than escape sequences on Windows systems.

MODEL_PATH = r"C:\Users\Rizwan computers\Desktop\AI-Project_1\student_stress_model.pkl"


# ============================================================
# LOAD MODEL FUNCTION
# ============================================================
# WHAT IS @st.cache_resource?
# Streamlit re-runs the entire Python script from line 1 every time a user interacts 
# with any UI widget (buttons, sliders, text boxes). Re-reading heavy files from disk 
# on every click causes slow performance.
# @st.cache_resource tells Streamlit to run this function ONCE, store the loaded model 
# object in system RAM, and reuse it instantly across all subsequent app interactions.

@st.cache_resource
def load_model():
    """
    Load the trained machine learning pipeline.

    The saved model already contains:
        StandardScaler
        +
        RandomForestClassifier

    Therefore, input data must NOT be manually scaled here.
    """

    # Check if the file path exists on the file system; if missing, prevent errors
    if not os.path.exists(MODEL_PATH):
        return None

    # Load and return the serialized model object pipeline using joblib
    return joblib.load(MODEL_PATH)


# Call the function to load the model into memory upon initial app execution
model = load_model()


# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================
# Safety check: If the model file could not be found or loaded, display a red error message 
# in the web app and halt further execution cleanly.

if model is None:

    # Display an error message box to the user on the screen
    st.error(
        f"""
        ❌ Model file not found.

        Please make sure that:

        `{MODEL_PATH}`

        is present.
        """
    )

    # Completely stop execution of the Streamlit script beyond this line
    st.stop()


# ============================================================
# EXPECTED FEATURES (Feature Order Enforcement)
# ============================================================
# CRITICAL ML CONCEPT:
# Machine Learning models do not understand real-world concepts; they operate purely on 
# ordered mathematical arrays. The column names and order supplied to model.predict() 
# MUST EXACTLY MATCH the exact order used when model.fit() was trained.

FEATURE_ORDER = [
    "sleep_hours",             # Feature 1: Average nightly sleep duration
    "study_hours_per_day",     # Feature 2: Daily academic study commitment
    "heart_rate_bpm",          # Feature 3: Physiological heart rate measurement
    "academic_workload",       # Feature 4: Subjective workload rating (1-10)
    "caffeine_intake_mg",      # Feature 5: Daily caffeine consumption in milligrams
    "physical_activity_min",   # Feature 6: Exercise or physical activity duration in minutes
    "social_interaction_hrs",  # Feature 7: Time spent socially interacting with peers/family
    "gpa",                     # Feature 8: Grade Point Average academic indicator
    "perceived_control"        # Feature 9: Sense of personal/academic environmental control
]


# ============================================================
# EXPECTED STRESS CLASSES
# ============================================================
# The target categories/labels that our classifier was trained to predict.

STRESS_CLASSES = [
    "No Stress",               # Target Class 0: Healthy balanced state
    "Low Stress",              # Target Class 1: Mild stress manageable level
    "Medium Stress",           # Target Class 2: Moderate stress requiring attention
    "High Stress"              # Target Class 3: Critical stress level requiring intervention
]


# ============================================================
# HEADER SECTION
# ============================================================
# Render decorative main app headers using custom HTML class definitions.

st.markdown(
    '<div class="main-title"> Student Stress Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine Learning Based Student Stress Level Prediction
    </div>
    """,
    unsafe_allow_html=True
)

# Draw a visual horizontal divider line across the page width
st.markdown("---")


# ============================================================
# SIDEBAR NAVIGATION & SYSTEM INFORMATION
# ============================================================
# Streamlit's 'with st.sidebar:' block anchors UI widgets into a dedicated 
# collapsible side-panel on the left-hand side of the application.

with st.sidebar:

    # Display section header in sidebar
    st.header(" About the System")

    # Explanatory write-up about how the AI system functions
    st.write(
        """
        This application uses a trained Random Forest machine
        learning model to predict a student's stress level based
        on lifestyle, academic, physical, and personal factors.
        """
    )

    # Horizontal divider line inside sidebar
    st.markdown("---")

    # Display subheader listing the four target classes
    st.subheader("Prediction Classes")

    # Bulleted display of stress levels
    st.write("• No Stress")
    st.write("• Low Stress")
    st.write("• Medium Stress")
    st.write("• High Stress")

    st.markdown("---")

    # Display subheader specifying underlying algorithm architecture
    st.subheader(" Model")

    st.write("Random Forest Classifier")

    st.markdown("---")

    # Small muted footer caption text in sidebar
    st.caption(
        "Academic ML Project | Student Stress Detection"
    )


# ============================================================
# MAIN INTRODUCTION SECTION
# ============================================================

st.subheader(" Enter Student Information")

st.write(
    """
    Please provide the student's information below. The trained
    machine learning model will use these nine input features to
    predict the student's stress level.
    """
)


# ============================================================
# HELPER FUNCTION FOR MANUAL INPUT VALIDATION
# ============================================================
# WHAT IS DATA VALIDATION & ERROR HANDLING?
# Users might accidentally type letters instead of numbers, or enter impossible numbers 
# (e.g. 50 hours of sleep per day). This custom Python function validates inputs safely, 
# ensuring user input is within valid bounds before passing it to the ML model.

def get_numeric_input(
    label,           # Screen text label explaining the required field
    min_value,       # Lowest permissible numerical value accepted
    max_value,       # Highest permissible numerical value accepted
    help_text,       # Tooltip text displayed on hovering over the field
    key,             # Unique identifier key used internally by Streamlit for widget state
    integer=False    # Boolean flag: True if whole numbers are required, False for decimals
):
    """
    Create a manual text input field with placeholder range hints and validate the entered value.
    """

    # Dynamically compose placeholder guidance text shown inside the empty text field
    placeholder_text = f"Range: {min_value} to {max_value}"

    # Render a Streamlit text input box and capture the user's raw typed input as a string
    value = st.text_input(
        label,
        value="",                     # Default field state is empty string
        placeholder=placeholder_text, # Display expected bounds inside text field
        help=help_text,               # Educational tooltip
        key=key                       # Unique widget identifier
    )

    # If user has not typed anything yet, return None to signal missing input
    if value.strip() == "":
        return None

    # Try block: Attempt to parse the typed text string into numerical float or integer format
    try:

        if integer:
            number = int(value)       # Convert raw string into an integer (whole number)
        else:
            number = float(value)     # Convert raw string into a float (decimal number)

    # Catch ValueError if user entered non-numeric text characters (e.g. "abc")
    except ValueError:

        # Display red inline error warning in the interface
        st.error(
            f"Please enter a valid number for **{label}**."
        )

        return None

    # Range check validation: verify if parsed number falls outside valid parameters
    if number < min_value or number > max_value:

        # Display range error message to user
        st.error(
            f"**{label}** must be between "
            f"{min_value} and {max_value}."
        )

        return None

    # Return successfully validated number
    return number


# ============================================================
# INPUT FORM CREATION
# ============================================================
# WHAT IS A STREAMLIT FORM (st.form)?
# Normally, Streamlit re-calculates the screen instantly every time a user types a single key.
# Wrapping input controls inside a 'form' stops updates until the user clicks the 
# "Predict Stress Level" submit button at the bottom.

with st.form("stress_prediction_form"):

    # --------------------------------------------------------
    # SECTION 1: SLEEP AND STUDY INPUTS
    # --------------------------------------------------------

    st.markdown("### Sleep & Study")

    # Divide screen into 2 parallel columns side-by-side
    col1, col2 = st.columns(2)

    # Place Sleep input in column 1
    with col1:

        sleep_hours = get_numeric_input(
            "Sleep Hours per Day",
            3.5,                       # Minimum sleep boundary
            9.5,                       # Maximum sleep boundary
            "Expected range: 3.5 to 9.5 hours",
            "sleep_hours_input",
            integer=False             # Accepts decimal values (e.g. 7.5 hours)
        )

    # Place Study input in column 2
    with col2:

        study_hours_per_day = get_numeric_input(
            "Study Hours per Day",
            1.0,                       # Minimum study boundary
            12.0,                      # Maximum study boundary
            "Expected range: 1.0 to 12.0 hours",
            "study_hours_input",
            integer=False             # Accepts decimal values (e.g. 4.5 hours)
        )


    # --------------------------------------------------------
    # SECTION 2: PHYSICAL FACTORS
    # --------------------------------------------------------

    st.markdown("### Physical Factors")

    # Split row into two equal visual columns
    col3, col4 = st.columns(2)

    # Heart Rate input in column 3
    with col3:

        heart_rate_bpm = get_numeric_input(
            "Heart Rate (BPM)",
            60,                        # Minimum physiological resting heart rate bound
            110,                       # Maximum physiological heart rate bound
            "Expected range: 60 to 110 BPM",
            "heart_rate_input",
            integer=True              # Expects whole integer numbers
        )

    # Physical Activity input in column 4
    with col4:

        physical_activity_min = get_numeric_input(
            "Physical Activity (Minutes per Day)",
            0,                         # Minimum minutes
            90,                        # Maximum minutes
            "Expected range: 0 to 90 minutes",
            "physical_activity_input",
            integer=True              # Expects whole integer numbers
        )


    # --------------------------------------------------------
    # SECTION 3: ACADEMIC FACTORS
    # --------------------------------------------------------

    st.markdown("### Academic Factors")

    # Split row into two columns
    col5, col6 = st.columns(2)

    # Render a graphical slider widget for workload scale (1 to 10) in column 5
    with col5:

        academic_workload = st.slider(
            "Academic Workload",
            min_value=1,              # Scale minimum
            max_value=10,             # Scale maximum
            value=5,                  # Initial default slider position
            step=1,                   # Step increment value
            help="1 = Very Low, 10 = Very High"
        )

    # GPA numeric input field in column 6
    with col6:

        gpa = get_numeric_input(
            "GPA",
            2.00,                      # Minimum valid GPA threshold
            4.00,                      # Maximum standard scale GPA
            "Expected range: 2.00 to 4.00",
            "gpa_input",
            integer=False             # Accepts decimal precision (e.g. 3.75)
        )


    # --------------------------------------------------------
    # SECTION 4: LIFESTYLE FACTORS
    # --------------------------------------------------------

    st.markdown("### Lifestyle Factors")

    # Split row into two columns
    col7, col8 = st.columns(2)

    # Caffeine Intake input field in column 7
    with col7:

        caffeine_intake_mg = get_numeric_input(
            "Caffeine Intake (mg per Day)",
            0,                         # Minimum caffeine intake
            600,                       # Maximum intake limit
            "Expected range: 0 to 600 mg",
            "caffeine_input",
            integer=True              # Integer representation
        )

    # Social Interaction input field in column 8
    with col8:

        social_interaction_hrs = get_numeric_input(
            "Social Interaction (Hours per Day)",
            0.2,                       # Minimum social hours boundary
            6.0,                       # Maximum social hours boundary
            "Expected range: 0.2 to 6.0 hours",
            "social_interaction_input",
            integer=False             # Decimal hours allowed (e.g. 1.5)
        )


    # --------------------------------------------------------
    # SECTION 5: PERSONAL PERCEIVED CONTROL
    # --------------------------------------------------------

    st.markdown("### Personal Control")

    # Render interactive slider for Perceived Control rating (1 to 10)
    perceived_control = st.slider(
        "Perceived Control",
        min_value=1,                  # Scale minimum
        max_value=10,                 # Scale maximum
        value=5,                      # Default slider starting position
        step=1,                       # Increment per slider tick
        help="1 = Very Low Control, 10 = Very High Control"
    )


    # Horizontal visual divider line within the form
    st.markdown("---")


    # --------------------------------------------------------
    # FORM SUBMISSION BUTTON
    # --------------------------------------------------------

    # Form action submit button; execution shifts below only when clicked
    submitted = st.form_submit_button(
        "🔍 Predict Stress Level",
        use_container_width=True      # Expands button width across full form width
    )


# ============================================================
# PREDICTION LOGIC & PIPELINE EXECUTION
# ============================================================
# This block triggers ONLY when the user clicks the "Predict Stress Level" submit button.

if submitted:

    # --------------------------------------------------------
    # STEP 1: VERIFY THAT NO REQUIRED FIELD WAS LEFT BLANK
    # --------------------------------------------------------

    missing_input = False             # Initialize boolean check variable

    # Validate each variable sequentially; if any return None, mark flag True
    if sleep_hours is None:
        missing_input = True

    if study_hours_per_day is None:
        missing_input = True

    if heart_rate_bpm is None:
        missing_input = True

    if physical_activity_min is None:
        missing_input = True

    if gpa is None:
        missing_input = True

    if caffeine_intake_mg is None:
        missing_input = True

    if social_interaction_hrs is None:
        missing_input = True


    # If any input is invalid or missing, display warning and block execution
    if missing_input:

        st.warning(
            "⚠️ Please enter all required values before making a prediction."
        )

        st.stop()                     # Prevent executing prediction logic with missing data


    # --------------------------------------------------------
    # STEP 2: CONVERT USER INPUTS INTO A PANDAS DATAFRAME
    # --------------------------------------------------------
    # Machine learning models require data arranged as a 2D matrix/table.
    # We convert the collected user values into a Pandas DataFrame row with exact feature column names.

    input_data = pd.DataFrame(
        [[
            sleep_hours,
            study_hours_per_day,
            heart_rate_bpm,
            academic_workload,
            caffeine_intake_mg,
            physical_activity_min,
            social_interaction_hrs,
            gpa,
            perceived_control
        ]],
        columns=FEATURE_ORDER          # Enforce exact column order from training step
    )


    # --------------------------------------------------------
    # STEP 3: RUN PREDICTION THROUGH ML PIPELINE
    # --------------------------------------------------------
    # Pass input_data DataFrame through model.predict(). The pipeline automatically handles
    # data standardization internally before passing transformed numerical values into 
    # the underlying Random Forest Classifier logic.

    try:

        # model.predict() returns an array of predictions; [0] grabs the single row prediction result
        prediction = model.predict(input_data)[0]

    except Exception as e:

        # Exception handling to capture runtime errors cleanly
        st.error(
            f"❌ Prediction error: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # STEP 4: DISPLAY PREDICTION RESULTS VISUALLY
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Result")

    # Convert returned prediction result to string text for safe evaluation
    prediction = str(prediction)


    # Color-coded conditional result boxes based on stress category severity
    if prediction == "No Stress":

        # Green success notification box for optimal non-stressed state
        st.success(
            f"### Predicted Stress Level: {prediction}"
        )

    elif prediction == "Low Stress":

        # Blue info callout box for mild stress levels
        st.info(
            f"### Predicted Stress Level: {prediction}"
        )

    elif prediction == "Medium Stress":

        # Yellow warning callout box for moderate elevated stress
        st.warning(
            f"### Predicted Stress Level: {prediction}"
        )

    elif prediction == "High Stress":

        # Red critical error callout box for high stress detection
        st.error(
            f"### Predicted Stress Level: {prediction}"
        )

    else:

        # Standard plain text fallback display
        st.write(
            f"### Predicted Stress Level: {prediction}"
        )


    # --------------------------------------------------------
    # STEP 5: DISPLAY SUMMARY OF SUBMITTED INPUT VALUES
    # --------------------------------------------------------
    # Displays the exact feature values provided by the user using Streamlit key-value metric widgets.

    st.markdown("---")

    st.subheader("Input Summary")

    # Divide summary metrics area across three equal visual columns
    summary_col1, summary_col2, summary_col3 = st.columns(3)


    # Column 1 Metrics: Sleep, Study, and Heart Rate
    with summary_col1:

        st.metric(
            "Sleep",
            f"{sleep_hours:.1f} hrs"    # Formats float value to 1 decimal place
        )

        st.metric(
            "Study",
            f"{study_hours_per_day:.1f} hrs"
        )

        st.metric(
            "Heart Rate",
            f"{heart_rate_bpm} BPM"
        )


    # Column 2 Metrics: Workload, Caffeine, Physical Activity
    with summary_col2:

        st.metric(
            "Academic Workload",
            academic_workload
        )

        st.metric(
            "Caffeine",
            f"{caffeine_intake_mg} mg"
        )

        st.metric(
            "Physical Activity",
            f"{physical_activity_min} min"
        )


    # Column 3 Metrics: Social Interaction, GPA, Perceived Control
    with summary_col3:

        st.metric(
            "Social Interaction",
            f"{social_interaction_hrs:.1f} hrs"
        )

        st.metric(
            "GPA",
            f"{gpa:.2f}"               # Formats float value to 2 decimal places
        )

        st.metric(
            "Perceived Control",
            perceived_control
        )


# ============================================================
# FOOTER / DISCLAIMER SECTION
# ============================================================

st.markdown("---")

# Muted caption disclaimer detailing project academic nature and dataset constraints
st.caption(
    """
    Disclaimer: This system is an academic machine learning project.
    The dataset used for training is synthetic/artificial and the
    prediction should not be considered a medical or psychological
    diagnosis.
    """
)