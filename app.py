from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from pykrige.ok import OrdinaryKriging


app = Flask(__name__)


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

                name = str(
                    row[0]
                )

                lon = float(
                    row[1]
                )

                lat = float(
                    row[2]
                )

                vs30 = float(
                    row[3]
                )


                if not (

                    np.isnan(lon)

                    or

                    np.isnan(lat)

                    or

                    np.isnan(vs30)

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


        # Remove duplicate TSMIP coordinates
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
# Clean borehole data
# =========================================================

def clean_borehole_data(df):

    valid_data = []


    for idx, row in df.iterrows():

        try:

            lon = float(
                row[1]
            )

            lat = float(
                row[2]
            )

            mean = float(
                row[3]
            )

            sd = float(
                row[4]
            )


            if not (

                np.isnan(lon)

                or

                np.isnan(lat)

                or

                np.isnan(mean)

                or

                np.isnan(sd)

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

    df = load_data()


    if df is None:

        return jsonify([])


    boreholes = []


    for idx, row in df.iterrows():

        try:

            lon_val = float(
                row[1]
            )

            lat_val = float(
                row[2]
            )

            mean_val = float(
                row[3]
            )

            sd_val = float(
                row[4]
            )

            hole_name = str(
                row[0]
            )


            if not (

                np.isnan(lon_val)

                or

                np.isnan(lat_val)

                or

                np.isnan(mean_val)

                or

                np.isnan(sd_val)

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
# Vs30 Prediction
# =========================================================

@app.route(
    '/predict',
    methods=['POST']
)
def predict():

    try:

        # =================================================
        # Target coordinates
        # =================================================

        data = request.get_json()


        if data is None:

            return jsonify({
                'error': 'Invalid request data.'
            }), 400


        target_lon = float(
            data['lon']
        )

        target_lat = float(
            data['lat']
        )


        # =================================================
        # Load borehole data
        # =================================================

        df = load_data()


        if df is None:

            return jsonify({
                'error': 'Borehole data file not found.'
            }), 400


        clean_np = clean_borehole_data(
            df
        )


        # Make sure enough valid data exist
        if (

            clean_np.ndim != 2

            or

            clean_np.shape[0] < 3

            or

            clean_np.shape[1] < 4

        ):

            return jsonify({
                'error':
                    'Insufficient valid borehole data.'
            }), 400


        x_known = clean_np[:, 0]

        y_known = clean_np[:, 1]

        mean_known = clean_np[:, 2]

        sd_known = clean_np[:, 3]


        # =================================================
        # STEP 1
        #
        # Borehole MCS Mean
        # ->
        # Base Mean Kriging
        # =================================================

        OK_mean = OrdinaryKriging(

            x_known,

            y_known,

            mean_known,

            variogram_model='spherical',

            verbose=False,

            enable_plotting=False

        )


        # Base Mean at target location
        base_mean_arr, _ = OK_mean.execute(

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
        #
        # Load TSMIP measured Vs30
        # =================================================

        tsmip_df = load_tsmip()


        if (

            tsmip_df is None

            or

            len(tsmip_df) < 3

        ):

            return jsonify({
                'error':
                    'Insufficient TSMIP data.'
            }), 400


        tsmip_lon = (

            tsmip_df[
                'lon'
            ]

            .to_numpy(
                dtype=float
            )

        )


        tsmip_lat = (

            tsmip_df[
                'lat'
            ]

            .to_numpy(
                dtype=float
            )

        )


        tsmip_measured = (

            tsmip_df[
                'vs30'
            ]

            .to_numpy(
                dtype=float
            )

        )


        # =================================================
        # STEP 3
        #
        # Calculate Base Mean at every TSMIP station
        # =================================================

        tsmip_base_arr, _ = OK_mean.execute(

            'points',

            tsmip_lon,

            tsmip_lat

        )


        tsmip_base = np.asarray(

            tsmip_base_arr,

            dtype=float

        ).ravel()


        # =================================================
        # STEP 4
        #
        # Residual
        #
        # Residual =
        # TSMIP measured Vs30
        # -
        # SPT-N / MCS Base Mean
        # =================================================

        residual = (

            tsmip_measured

            -

            tsmip_base

        )


        # =================================================
        # STEP 5
        #
        # TSMIP Residual Kriging
        # =================================================

        OK_residual = OrdinaryKriging(

            tsmip_lon,

            tsmip_lat,

            residual,

            variogram_model='spherical',

            verbose=False,

            enable_plotting=False

        )


        residual_arr, _ = OK_residual.execute(

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
        # STEP 6
        #
        # Final Mean
        #
        # Final Mean =
        # Base Mean
        # +
        # TSMIP Residual Correction
        # =================================================

        final_mean = (

            base_mean

            +

            residual_correction

        )


        # =================================================
        # STEP 7
        #
        # MCS SD Spatial Interpolation
        #
        # IMPORTANT:
        # Kriging variance is NOT added.
        # =================================================

        OK_sd = OrdinaryKriging(

            x_known,

            y_known,

            sd_known,

            variogram_model='spherical',

            verbose=False,

            enable_plotting=False

        )


        sd_arr, _ = OK_sd.execute(

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


        # SD cannot be negative
        final_sd = max(
            final_sd,
            0.0
        )


        # =================================================
        # Final validation
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
            f"Target: "
            f"({target_lon}, {target_lat})"
        )


        print(
            f"SPT-N Base Mean = "
            f"{base_mean:.2f} m/s"
        )


        print(
            f"TSMIP Residual Correction = "
            f"{residual_correction:+.2f} m/s"
        )


        print(
            f"Final Vs30 Mean = "
            f"{final_mean:.2f} m/s"
        )


        print(
            f"MCS SD = "
            f"{final_sd:.2f} m/s"
        )


        # =================================================
        # Return results to frontend
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
# Start Flask
# =========================================================

if __name__ == '__main__':

    print(
        "\nServer started successfully!\n"
    )

    app.run(
        debug=True,
        port=5000
    )
