from datetime import datetime
from model import encode_data, load_model, predict_model
from preprocess import preprocess_input_data
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import plotly
import plotly.express as px

app = Flask(__name__, template_folder="templates")

# Load the model on startup
#
# Load existing data (or create an empty DataFrame with columns)
file_path = "accident_data_to_predict.csv"
file_path_accident_data = "data/accident_data.csv"
weekday_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    data = pd.DataFrame(
        columns=[
            "Newspaper Name",
            "Accident date",
            "Header",
            "News title",
            "Deaths",
            "Injured",
        ]
    )


@app.route("/")
def home():
    max_date = datetime.now().date()  # Get today's date
    return render_template("form.html", max_date=max_date)


@app.route("/submit", methods=["POST"])
def submit():
    # Capture form data
    newspaper_name = request.form["newspaper_name"]
    accident_date = request.form["accident_date"]
    header = request.form["header"]
    news_title = request.form["news_title"]
    deaths = request.form["deaths"]
    injured = request.form["injured"]
    # Append new data to the DataFrame
    new_data = pd.DataFrame(
        [[newspaper_name, accident_date, header, news_title, deaths, injured]],
        columns=[
            "Newspaper Name",
            "Accident date",
            "Header",
            "News title",
            "Deaths",
            "Injured",
        ],
    )
    global data
    data = pd.concat([data, new_data], ignore_index=True)

    # Save the updated data back to CSV
    data.to_csv(file_path, index=False)
    return redirect(url_for("predict"))


@app.route("/predict")
def predict():
    # model = train_model()
    data = pd.read_csv("accident_data_to_predict.csv")
    # Preprocess the input data
    global processed_data
    processed_data = preprocess_input_data(data)
    encoded_df = encode_data(processed_data)
    X = encoded_df.drop(columns=["Severity"])
    # Make a prediction
    model = load_model()
    prediction = predict_model(model, X)
    # data.to_csv('pred_accident_data_to_predict.csv', index=False)
    # Return the result
    data["Predicted Severity"] = prediction

    X["Predicted Severity"] = prediction

    # Save the updated DataFrame to a new CSV
    # X.to_csv('output_cleaned_csv.csv', index=False)
    # Save the updated DataFrame to a new CSV
    data.to_csv("output_original_csv.csv", index=False)
    # data = pd.read_csv("output_original_csv.csv")
    print(data.columns)
    table = (
        data.drop(columns=["Severity"]).tail(1).to_html(classes="table table-striped")
    )
    return render_template("result.html", prediction=table[:-1])
    # return jsonify({'prediction': prediction})


def create_plot(trend_type):
    # Create a bar chart of accidents by month
    input_df = pd.read_csv(file_path_accident_data)
    processed_data = preprocess_input_data(input_df)
    monthly_trends = (
        processed_data.groupby(["Year", "Month"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="AccidentCount")
    )
    monthly_trends_pivot = monthly_trends.pivot(
        index="Month", columns="Year", values="AccidentCount"
    )

    if trend_type == "seasonal":
        trend_data = processed_data.groupby("Season").size().reset_index(name="AccidentCount")
        fig = px.bar(
            trend_data, x="Season", y="AccidentCount", title="Seasonal Accident Trends"
        )
        fig.update_layout(xaxis_title="Season", yaxis_title="Number of Accidents")
        
    elif trend_type == "monthly_trend_year":
        # Aggregate the data by month to analyze accident patterns over time
        monthly_accidents = processed_data.groupby('Month-Year').size().rename("Monthly Accidents")
        # Convert to DataFrame for easier plotting with Seaborn
        monthly_accidents_df = monthly_accidents.reset_index()
        monthly_accidents_df['Month-Year'] = monthly_accidents_df['Month-Year'].dt.to_timestamp()
        
        fig = px.line(monthly_accidents_df, x='Month-Year', y='Monthly Accidents',
              title="Monthly Road Accident Patterns",
              labels={'Month-Year': 'Date', 'Monthly Accidents': 'Accident Count'})
        fig.update_layout(xaxis_title="Period", yaxis_title="Accident Count")
        
        
    elif trend_type == "monthly_trend":
        fig = go.Figure()
        for year in monthly_trends_pivot.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends_pivot.index,
                    y=monthly_trends_pivot[year],
                    mode="lines+markers",
                    name=str(year),
                )
            )

        fig.update_layout(
            title="Monthly Accident Trends Over Time",
            xaxis=dict(
                title="Month",
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=[
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            ),
            yaxis_title="Number of Accidents",
        )
    elif trend_type == "yearly":
        trend_data = processed_data.groupby("Year").size().reset_index(name="AccidentCount")
        fig = px.line(
            trend_data, x="Year", y="AccidentCount", title="Yearly Accident Trends"
        )
        fig.update_layout(xaxis_title="Year", yaxis_title="Number of Accidents")
        # Convert the Plotly figure to JSON
    elif trend_type == "severity":
        trend_data = processed_data.groupby("Severity").size().reset_index(name="AccidentCount")
        fig = px.bar(
            trend_data, x="Severity", y="AccidentCount", title="Accidents by Severity"
        )
        fig.update_layout(xaxis_title="Severity", yaxis_title="Number of Accidents")

    elif trend_type == "weekday":
       
        trend_data = processed_data.groupby('Weekday').size().reindex(weekday_mapping.values()).reset_index(name='AccidentCount')
        #trend_data = processed_data.groupby("Weekday").size().reset_index(name="AccidentCount")
        fig = px.bar(
            trend_data,
            x="Weekday",
            y="AccidentCount",
            title="Accident Trends by Weekday",
        )
        fig.update_layout(xaxis_title="Weekday", yaxis_title="Number of Accidents")
    elif trend_type == "weekday_severity":
        trend_data = (
            processed_data.groupby(["Weekday", "Severity"])
            .size()
            
            .reset_index(name="AccidentCount")
        )
        fig = px.bar(
            trend_data,
            x="Weekday",
            y="AccidentCount",
            color="Severity",
            category_orders = {'Weekday' : ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
            title="Accident Severity by Weekday",
            barmode="group",
        )
        fig.update_layout(xaxis_title="Weekday", yaxis_title="Number of Accidents")

    elif trend_type == "top10":
        trend_data = processed_data["Identified Location"].value_counts().head(10).reset_index()
        trend_data.columns = ["Identified Location", "AccidentCount"]
        fig = px.bar(
            trend_data,
            x="Identified Location",
            y="AccidentCount",
            title="Top 10 Locations for Accidents",
        )
        fig.update_layout(xaxis_title="Location", yaxis_title="Number of Accidents")
    elif trend_type == "vehicle_type":
        trend_data = processed_data["Car Type"].value_counts().reset_index()
        trend_data.columns = ["Car Type", "AccidentCount"]
        fig = px.bar(
            trend_data,
            x="Car Type",
            y="AccidentCount",
            title="Accidents by Vehicle Type",
        )
        fig.update_layout(xaxis_title="Vehicle Type", yaxis_title="Number of Accidents")
    elif trend_type == "yearlytop10":
        # Filter data for the selected year and find the top 10 locations
        trend_data = (
            processed_data["Identified Location"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        trend_data.columns = ["Identified Location", "AccidentCount"]
        fig = px.bar(
            trend_data,
            x="Identified Location",
            y="AccidentCount",
            title=f"Top 10 Accident Locations",
        )
        fig.update_layout(xaxis_title="Location", yaxis_title="Number of Accidents")

    else:  # Default to weekday trend if trend_type is not recognized
        trend_data = processed_data.groupby("Weekday").size().reindex(weekday_mapping.values()).reset_index(name="AccidentCount")
        fig = px.bar(
            trend_data,
            x="Weekday",
            y="AccidentCount",
            title="Accident Trends by Weekday",
        )
        fig.update_layout(xaxis_title="Weekday", yaxis_title="Number of Accidents")

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json


@app.route("/visualise")
def visualise():
    trend_type = request.args.get("trend_type", "weekday")
    # Create Bar chart
    plot_json = create_plot(trend_type)
    # Use render_template to pass graphJSON to html
    return render_template("visualise.html", plot_json=plot_json,selected_trend=trend_type)


""" @app.route('/train')
def train():
    # Train a new model
    model = train_model()
        #print(classification_report)

    return jsonify({'message': 'Model trained successfully'})

 """


## run the app
if __name__ == "__main__":
    app.run(debug=True)
