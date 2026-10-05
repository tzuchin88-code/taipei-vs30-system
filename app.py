from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from pykrige.ok import OrdinaryKriging

app = Flask(__name__)


# =========================================================
# Global variables
# Kriging models will be created only once
# =========================================================

BOREHOLE_DF = None

OK_MEAN = None
OK_RESIDUAL = None
OK_SD = None

MODEL_READY = False
MODEL_ERROR = None


# =========================================================
# Load borehole data
#
# A = Borehole name
# B = Longitude
# C = Latitude
# D = MCS Mean
# E = MCS SD
# =========================================================

def load_data():

    for fname in [
        'ALL_VS30.xlsx',
        'ALL_VS30.xlsm',
        '8孔.xlsx',
        '8孔.xlsm'
    ]:

        if os.path.exists(fname):

            try:

                df = pd.read_excel(
                    fname,
                    header=None
                )

                # Remove duplicate borehole names
                df = df.drop_duplicates(
                    subset=[0],
                    keep='first'
                )

                # Remove duplicate coordinates
                df = df.drop_duplicates(
                    subset=[1, 2],
                    keep='first'
                )

                print(
                    f"Loaded borehole data: {fname}, "
                    f"effective boreholes: {len(df)}"
                )

                return df

            except Exception as e:

                print(
                    f"Failed to read {fname}: {e}"
                )

    return None


# =========================================================
# Clean borehole data
# =========================================================

def clean_borehole_data(df):

    valid_data = []

    for idx, row in df.iterrows():

        try:

            lon = float(row[1])
            lat = float(row[2])
            mean = float(row[3])
            sd = float(row[4])

            if not (
                np.isnan(lon)
                or np.isnan(lat)
                or np.isnan(mean)
                or np.isnan(sd)
            ):

                valid_data.append([
                    lon,
                    lat,
                    mean,
                    sd
                ])

        except (ValueError, TypeError):

            continue

    return np.array(
        valid_data,
        dtype=float
    )


# =========================================================
# Load TSMIP measured Vs30
#
# A = Station name
# B = Longitude
# C = Latitude
# D = Measured Vs30
# =========================================================

def load_tsmip():

    fname = 'TSMIP_VS30.xlsx'

    if not os.path.exists(fname):

        print(
            "TSMIP_VS30.xlsx not found."
        )

        return None

    try:

        df = pd.read_excel(
            fname,
            header=None
        )

        valid_data = []

        for idx, row in df.iterrows():

            try:

                name = str(row[0])
                lon = float(row[1])
                lat = float(row[2])
                vs30 = float(row[3])

                if not (
                    np.isnan(lon)
                    or np.isnan(lat)
                    or np.isnan(vs30)
                ):

                    valid_data.append([
                        name,
                        lon,
                        lat,
                        vs30
                    ])

            except (ValueError, TypeError):

                continue

        tsmip_df = pd.DataFrame(
            valid_data,
            columns=[
                'name',
                'lon',
                'lat',
                'vs30'
            ]
        )

        # Remove duplicate coordinates
        tsmip_df = tsmip_df.drop_duplicates(
            subset=[
                'lon',
                'lat'
            ],
            keep='first'
        )

        print(
            f"Loaded TSMIP data: "
            f"{len(tsmip_df)} stations"
        )

        return tsmip_df

    except Exception as e:

        print(
            f"Failed to read TSMIP data: {e}"
        )

        return None


# =========================================================
# Build all Kriging models
#
# IMPORTANT:
# This function runs only once when the server starts.
# =========================================================

def build_models():

    global BOREHOLE_DF
    global OK_MEAN
    global OK_RESIDUAL
    global OK_SD
    global MODEL_READY
    global MODEL_ERROR

    try:

        print("")
        print("==========================================")
        print("Building Vs30 Kriging models...")
        print("==========================================")


        # -------------------------------------------------
        # 1. Borehole data
        # -------------------------------------------------

        BOREHOLE_DF = load_data()

        if BOREHOLE_DF is None:

            raise RuntimeError(
                "Borehole data file not found."
            )


        clean_np = clean_borehole_data(
            BOREHOLE_DF
        )


        if (
            clean_np.ndim != 2
            or clean_np.shape[0] < 3
            or clean_np.shape[1] < 4
        ):

            raise RuntimeError(
                "Insufficient valid borehole data."
            )


        x_known = clean_np[:, 0]
        y_known = clean_np[:, 1]

        mean_known = clean_np[:, 2]
        sd_known = clean_np[:, 3]


        print(
            f"Valid boreholes: "
            f"{len(x_known)}"
        )


        # =================================================
        # 2. Build Base Mean Kriging model
        # =================================================

        print(
            "Building Base Mean Kriging model..."
        )


        OK_MEAN = OrdinaryKriging(

            x_known,
            y_known,
            mean_known,

            variogram_model='spherical',

            verbose=False,
            enable_plotting=False

        )


        print(
            "Base Mean Kriging model ready."
        )


        # =================================================
        # 3. Load TSMIP
        # =================================================

        tsmip_df = load_tsmip()


        if (
            tsmip_df is None
            or len(tsmip_df) < 3
        ):

            raise RuntimeError(
                "Insufficient TSMIP data."
            )


        tsmip_lon = (
            tsmip_df['lon']
            .to_numpy(dtype=float)
        )


        tsmip_lat = (
            tsmip_df['lat']
            .to_numpy(dtype=float)
        )


        tsmip_measured = (
            tsmip_df['vs30']
            .to_numpy(dtype=float)
        )


        # =================================================
        # 4. Calculate Base Mean at TSMIP stations
        # =================================================

        print(
            "Calculating Base Mean at TSMIP stations..."
        )


        tsmip_base_arr, _ = OK_MEAN.execute(

            'points',

            tsmip_lon,

            tsmip_lat

        )


        tsmip_base = np.asarray(

            tsmip_base_arr,

            dtype=float

        ).ravel()


        # =================================================
        # 5. Calculate TSMIP residual
        #
        # Residual =
        # Measured TSMIP Vs30
        # -
        # Borehole Base Mean
        # =================================================

        residual = (

            tsmip_measured

            -

            tsmip_base

        )


        print(
            f"Residual Mean = "
            f"{np.mean(residual):.2f} m/s"
        )


        print(
            f"Residual Min = "
            f"{np.min(residual):.2f} m/s"
        )


        print(
            f"Residual Max = "
            f"{np.max(residual):.2f} m/s"
        )


        # =================================================
        # 6. Build Residual Kriging model
        # =================================================

        print(
            "Building TSMIP Residual Kriging model..."
        )


        OK_RESIDUAL = OrdinaryKriging(

            tsmip_lon,
            tsmip_lat,
            residual,

            variogram_model='spherical',

            verbose=False,
            enable_plotting=False

        )


        print(
            "TSMIP Residual Kriging model ready."
        )


        # =================================================
        # 7. Build MCS SD Kriging model
        #
        # Kriging variance is NOT added.
        # =================================================

        print(
            "Building MCS SD Kriging model..."
        )


        OK_SD = OrdinaryKriging(

            x_known,
            y_known,
            sd_known,

            variogram_model='spherical',

            verbose=False,
            enable_plotting=False

        )


        print(
            "MCS SD Kriging model ready."
        )


        # =================================================
        # Models ready
        # =================================================

        MODEL_READY = True
        MODEL_ERROR = None


        print("")
        print("==========================================")
        print("All Vs30 models are ready.")
        print("==========================================")
        print("")


    except Exception as e:

        MODEL_READY = False

        MODEL_ERROR = (
            f"{type(e).__name__}: {str(e)}"
        )


        print("")
        print("==========================================")
        print(
            f"MODEL BUILD ERROR: "
            f"{MODEL_ERROR}"
        )
        print("==========================================")
        print("")


# =========================================================
# Home
# =========================================================

@app.route('/')
def home():

    return render_template(
        'index.html'
    )


# =========================================================
# Send borehole locations to frontend
# =========================================================

@app.route(
    '/api/boreholes',
    methods=['GET']
)
def get_boreholes():

    global BOREHOLE_DF

    # Normally already loaded during startup.
    # Fallback only if necessary.
    if BOREHOLE_DF is None:

        BOREHOLE_DF = load_data()


    if BOREHOLE_DF is None:

        return jsonify([])


    boreholes = []


    for idx, row in BOREHOLE_DF.iterrows():

        try:

            lon_val = float(row[1])
            lat_val = float(row[2])

            mean_val = float(row[3])
            sd_val = float(row[4])

            hole_name = str(row[0])


            if not (
                np.isnan(lon_val)
                or np.isnan(lat_val)
                or np.isnan(mean_val)
                or np.isnan(sd_val)
            ):

                boreholes.append({

                    'name':
                        hole_name,

                    'lon':
                        lon_val,

                    'lat':
                        lat_val,

                    'mean':
                        mean_val,

                    'sd':
                        sd_val

                })


        except (ValueError, TypeError):

            continue


    print(
        f"Sent {len(boreholes)} boreholes to frontend."
    )


    return jsonify(
        boreholes
    )


# =========================================================
# Model status
#
# Optional diagnostic endpoint:
# /api/model-status
# =========================================================

@app.route(
    '/api/model-status',
    methods=['GET']
)
def model_status():

    return jsonify({

        'ready':
            MODEL_READY,

        'error':
            MODEL_ERROR

    })


# =========================================================
# Vs30 Prediction
#
# IMPORTANT:
# No Kriging models are rebuilt here.
# Only prediction is performed.
# =========================================================

@app.route(
    '/predict',
    methods=['POST']
)
def predict():

    try:

        # -------------------------------------------------
        # Check model status
        # -------------------------------------------------

        if not MODEL_READY:

            return jsonify({

                'error':
                    'Kriging models are not ready.',

                'detail':
                    MODEL_ERROR

            }), 503


        # -------------------------------------------------
        # Target coordinates
        # -------------------------------------------------

        data = request.get_json()


        if data is None:

            return jsonify({

                'error':
                    'Invalid request data.'

            }), 400


        target_lon = float(
            data['lon']
        )


        target_lat = float(
            data['lat']
        )


        # =================================================
        # STEP 1
        # Base Mean prediction
        # =================================================

        base_mean_arr, _ = OK_MEAN.execute(

            'points',

            np.array([
                target_lon
            ]),

            np.array([
                target_lat
            ])

        )


        base_mean = float(

            np.asarray(
                base_mean_arr
            ).ravel()[0]

        )


        # =================================================
        # STEP 2
        # TSMIP Residual Correction
        # =================================================

        residual_arr, _ = OK_RESIDUAL.execute(

            'points',

            np.array([
                target_lon
            ]),

            np.array([
                target_lat
            ])

        )


        residual_correction = float(

            np.asarray(
                residual_arr
            ).ravel()[0]

        )


        # =================================================
        # STEP 3
        # Final Vs30 Mean
        #
        # Final Mean =
        # Base Mean
        # +
        # Residual Correction
        # =================================================

        final_mean = (

            base_mean

            +

            residual_correction

        )


        # =================================================
        # STEP 4
        # MCS SD
        #
        # No Kriging variance is added.
        # =================================================

        sd_arr, _ = OK_SD.execute(

            'points',

            np.array([
                target_lon
            ]),

            np.array([
                target_lat
            ])

        )


        final_sd = float(

            np.asarray(
                sd_arr
            ).ravel()[0]

        )


        final_sd = max(
            final_sd,
            0.0
        )


        # =================================================
        # Validation
        # =================================================

        if not np.isfinite(
            base_mean
        ):

            raise ValueError(
                "Base Mean is not finite."
            )


        if not np.isfinite(
            residual_correction
        ):

            raise ValueError(
                "Residual correction is not finite."
            )


        if not np.isfinite(
            final_mean
        ):

            raise ValueError(
                "Final Vs30 Mean is not finite."
            )


        if not np.isfinite(
            final_sd
        ):

            raise ValueError(
                "Final Vs30 SD is not finite."
            )


        # =================================================
        # Console output
        # =================================================

        print(
            f"Prediction | "
            f"Lon={target_lon:.6f}, "
            f"Lat={target_lat:.6f} | "
            f"Base={base_mean:.2f} | "
            f"Correction={residual_correction:+.2f} | "
            f"Final={final_mean:.2f} | "
            f"SD={final_sd:.2f}"
        )


        # =================================================
        # Return result
        # =================================================

        return jsonify({

            'vs30_mean':
                final_mean,

            'vs30_sd':
                final_sd,

            'base_mean':
                base_mean,

            'tsmip_correction':
                residual_correction

        })


    except Exception as e:

        print(
            f"Prediction error: "
            f"{type(e).__name__}: {e}"
        )


        return jsonify({

            'error':
                f"{type(e).__name__}: {str(e)}"

        }), 500


# =========================================================
# Build models when app starts
# =========================================================

build_models()


# =========================================================
# Local development
# =========================================================

if __name__ == '__main__':

    print(
        "\nServer started successfully!\n"
    )

    app.run(
        debug=True,
        port=5000
    )
