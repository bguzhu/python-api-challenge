# python-api-challenge
Module 6 Challenge for Python using weather api

Weatherpy code


{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WeatherPy\n",
    "\n",
    "---\n",
    "\n",
    "## Starter Code to Generate Random Geographic Coordinates and a List of Cities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "!pip install citipy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Specify necessary dependencies\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import requests\n",
    "import time\n",
    "from scipy.stats import linregress\n",
    "\n",
    "#Get unique weather api key from the api_keys python file\n",
    "from api_keys import weather_api_key\n",
    "\n",
    "# Import citipy after pip install to find out the cities longitude and latitude coordinatesx\n",
    "from citipy import citipy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Generate the Cities List by Using the `citipy` Library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Empty list for lat and lon data\n",
    "lat_lngs = []\n",
    "\n",
    "# Empty list for all the cities\n",
    "cities = []\n",
    "\n",
    "# Range of values for longitude and latitude\n",
    "lat_range = (-90, 90)\n",
    "lng_range = (-180, 180)\n",
    "\n",
    "# Create random longitude and latitude\n",
    "lats = np.random.uniform(lat_range[0], lat_range[1], size=1500)\n",
    "lngs = np.random.uniform(lng_range[0], lng_range[1], size=1500)\n",
    "lat_lngs = zip(lats, lngs)\n",
    "\n",
    "#For every latitude and longitude in the list\n",
    "for lat_lng in lat_lngs:\n",
    "    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name\n",
    "    \n",
    "    #for each city, add to the cities list\n",
    "    if city not in cities:\n",
    "        cities.append(city)\n",
    "\n",
    "#print the total number of cities\n",
    "print(f\"Number of cities in the list: {len(cities)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cities"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Requirement 1: Create Plots to Showcase the Relationship Between Weather Variables and Latitude\n",
    "\n",
    "### Use the OpenWeatherMap API to retrieve weather data from the cities list generated in the started code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = f\"http://api.openweathermap.org/data/2.5/weather?units=metric&appid={weather_api_key}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define base url with unique weather api key\n",
    "url = f\"http://api.openweathermap.org/data/2.5/weather?units=metric&appid={weather_api_key}\"\n",
    "\n",
    "#Create an empty list to get the weather data for each city\n",
    "city_data = []\n",
    "\n",
    "#Print to tell the user that data is being gathered\n",
    "print(\"Beginning Data Retrieval     \")\n",
    "print(\"-----------------------------\")\n",
    "\n",
    "#Create counters for record and set\n",
    "record_count = 1\n",
    "set_count = 1\n",
    "\n",
    "#Loop through all the cities in city_data to get weather data\n",
    "for i, city in enumerate(cities):\n",
    "        \n",
    "    #Group cities in groups of 50 each\n",
    "    if (i % 50 == 0 and i >= 50):\n",
    "        set_count += 1\n",
    "        record_count = 0\n",
    "\n",
    "    #Create endpoint URL for each city\n",
    "    city_url = f\"{url}&q={city}\"\n",
    "    \n",
    "    #Keep track the url, record, and set numbers\n",
    "    print(\"Processing Record %s of Set %s | %s\" % (record_count, set_count, city))\n",
    "\n",
    "    #Increment the record count by 1\n",
    "    record_count += 1\n",
    "\n",
    "    #Run an API request/call for each of the cities\n",
    "    try:\n",
    "        #Get the request and convert to json to get the raw data\n",
    "        city_weather = requests.get(city_url).json()\n",
    "\n",
    "        #Take the latitude, longitude, max temp, humidity, cloudiness, wind speed, country, and date\n",
    "        city_lat = city_weather[\"coord\"][\"lat\"]\n",
    "        city_lng = city_weather[\"coord\"][\"lon\"]\n",
    "        city_max_temp = city_weather[\"main\"][\"temp_max\"]\n",
    "        city_humidity = city_weather[\"main\"][\"humidity\"]\n",
    "        city_clouds = city_weather[\"clouds\"][\"all\"]\n",
    "        city_wind = city_weather[\"wind\"][\"speed\"]\n",
    "        city_country = city_weather[\"sys\"][\"country\"]\n",
    "        city_date = city_weather[\"dt\"]\n",
    "\n",
    "        #Append the previous values into the city_data list\n",
    "        city_data.append({\"City\": city, \n",
    "                          \"Lat\": city_lat, \n",
    "                          \"Lng\": city_lng, \n",
    "                          \"Max Temp\": city_max_temp,\n",
    "                          \"Humidity\": city_humidity,\n",
    "                          \"Cloudiness\": city_clouds,\n",
    "                          \"Wind Speed\": city_wind,\n",
    "                          \"Country\": city_country,\n",
    "                          \"Date\": city_date})\n",
    "\n",
    "    #If an error is experienced, skip the city\n",
    "    except:\n",
    "        print(\"City not found. Skipping...\")\n",
    "        pass\n",
    "              \n",
    "#Indicate that Data Loading is complete \n",
    "print(\"-----------------------------\")\n",
    "print(\"Data Retrieval Complete      \")\n",
    "print(\"-----------------------------\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convert the cities weather data into a Pandas DataFrame\n",
    "city_data_df = pd.DataFrame(city_data)\n",
    "\n",
    "#Show the count\n",
    "city_data_df.count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Show the dataframe for the first five rows\n",
    "city_data_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Export the city data dataframe to a csv file named cities.csv\n",
    "city_data_df.to_csv(\"output_data/cities.csv\", index_label=\"City_ID\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read saved data\n",
    "city_data_df = pd.read_csv(\"output_data/cities.csv\", index_col=\"City_ID\")\n",
    "\n",
    "# Display sample data\n",
    "city_data_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create the Scatter Plots Requested\n",
    "\n",
    "#### Latitude Vs. Temperature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Create a scatter plot for latitude vs. temperature\n",
    "plt.scatter(\n",
    "    city_data_df[\"Lat\"],\n",
    "    city_data_df[\"Max Temp\"],\n",
    "    edgecolor=\"red\",\n",
    "    linewidths=1,\n",
    "    marker=\"o\",\n",
    "    alpha=0.8,\n",
    "    label=\"Cities\")\n",
    "\n",
    "#Make a title, x label title, and y label title\n",
    "plt.title(f\"City Max Latitude vs. Temperature\")\n",
    "plt.xlabel(\"Latitude\")\n",
    "plt.ylabel(\"Max Temp (C)\")\n",
    "\n",
    "#Save the plot to Fig1.png\n",
    "plt.savefig(\"output_data/Fig1.png\")\n",
    "\n",
    "# Show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Latitude Vs. Humidity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a scatter plot for latitude vs. humidity\n",
    "plt.scatter(\n",
    "    city_data_df[\"Lat\"],\n",
    "    city_data_df[\"Humidity\"],\n",
    "    edgecolor=\"green\",\n",
    "    linewidths=1,\n",
    "    marker=\"o\",\n",
    "    alpha=0.8,\n",
    "    label=\"Cities\")\n",
    "\n",
    "#Make a title, x label title, and y label title\n",
    "plt.title(f\"City Max Latitude vs. Humidity\")\n",
    "plt.xlabel(\"Latitude\")\n",
    "plt.ylabel(\"Humidity (%)\")\n",
    "\n",
    "#Save the plot to Fig2.png\n",
    "plt.savefig(\"output_data/Fig2.png\")\n",
    "\n",
    "# Show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Latitude Vs. Cloudiness"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a scatter plot for latitude vs. cloudiness\n",
    "plt.scatter(\n",
    "    city_data_df[\"Lat\"],\n",
    "    city_data_df[\"Cloudiness\"],\n",
    "    edgecolor=\"orange\",\n",
    "    linewidths=1,\n",
    "    marker=\"o\",\n",
    "    alpha=0.8,\n",
    "    label=\"Cities\")\n",
    "\n",
    "#Make a title, x label title, and y label title\n",
    "plt.title(f\"City Max Latitude vs. Cloudiness\")\n",
    "plt.xlabel(\"Latitude\")\n",
    "plt.ylabel(\"Cloudiness (%)\")\n",
    "\n",
    "# Save the plot to Fig3.png\n",
    "plt.savefig(\"output_data/Fig3.png\")\n",
    "\n",
    "# Show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Latitude vs. Wind Speed Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a scatter plot for latitude vs. wind speed\n",
    "plt.scatter(\n",
    "    city_data_df[\"Lat\"],\n",
    "    city_data_df[\"Wind Speed\"],\n",
    "    edgecolor=\"purple\",\n",
    "    linewidths=1,\n",
    "    marker=\"o\",\n",
    "    alpha=0.8,\n",
    "    label=\"Cities\")\n",
    "\n",
    "#Make a title, x label title, and y label title\n",
    "plt.title(f\"City Max Latitude vs. Wind Speed\")\n",
    "plt.xlabel(\"Latitude\")\n",
    "plt.ylabel(\"Wind Speed (%)\")\n",
    "\n",
    "# Save the plot to Fig4.png\n",
    "plt.savefig(\"output_data/Fig4.png\")\n",
    "\n",
    "# Show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "## Requirement 2: Compute Linear Regression for Each Relationship\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define a function to make the Linear Regression plots\n",
    "def plot_linear_regression(x_value, y_value, title, text_coord):\n",
    "    \n",
    "    (slope, intercept, rvalue, pvalue, stderr)= linregress(x_value, y_value)\n",
    "    regress_value = x_value * slope + intercept\n",
    "    line_eq = f\"y = {round(slope, 2)}x + {round(intercept, 2)}\"\n",
    "    \n",
    "    plt.scatter(x_value, y_value)\n",
    "    plt.plot(x_value, regress_value, \"r-\")\n",
    "    plt.annotate(line_eq, text_coord, fontsize=15, color=\"red\")\n",
    "    plt.xlabel(\"Lat\")\n",
    "    plt.ylabel(title)\n",
    "    print(f\"The r-value is {rvalue ** 2}\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a DataFrame with the Northern Hemisphere data (Latitude >= 0)\n",
    "northern_hemi_df = city_data_df[city_data_df[\"Lat\"] >=0]\n",
    "\n",
    "#Display the data\n",
    "northern_hemi_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a DataFrame with the Southern Hemisphere data (Latitude < 0)\n",
    "southern_hemi_df = city_data_df[city_data_df[\"Lat\"]<0]\n",
    "\n",
    "#Display the data\n",
    "southern_hemi_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###  Temperature vs. Latitude Linear Regression Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Northern Hemisphere\n",
    "x_values= northern_hemi_df[\"Lat\"]\n",
    "y_values= northern_hemi_df[\"Max Temp\"]\n",
    "plot_linear_regression(x_values, y_values, \"Max Temp\", (-40, 30))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Linear regression on Southern Hemisphere\n",
    "x_values= southern_hemi_df[\"Lat\"]\n",
    "y_values= southern_hemi_df[\"Max Temp\"]\n",
    "plot_linear_regression(x_values, y_values, \"Max Temp\", (-55, 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Discussion about the linear relationship:** YOUR RESPONSE HERE\n",
    "The regression linear graphs show that as latitude increases for the Northern Hemisphere, the max temperature goes down. For the Southern Hemisphere, when latitude increases, the max temperature goes up."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Humidity vs. Latitude Linear Regression Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Northern Hemisphere\n",
    "x_values= northern_hemi_df[\"Lat\"]\n",
    "y_values= northern_hemi_df[\"Humidity\"]\n",
    "plot_linear_regression(x_values, y_values, \"Humidity\", (50, 20))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Southern Hemisphere\n",
    "x_values= southern_hemi_df[\"Lat\"]\n",
    "y_values= southern_hemi_df[\"Humidity\"]\n",
    "plot_linear_regression(x_values, y_values, \"Humidity\", (-50, 30))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Discussion about the linear relationship:** YOUR RESPONSE HERE\n",
    "The regression linear graphs show that the humidity remains relatively constant for any value of latitude for both the Northern and Southern Hemispheres."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cloudiness vs. Latitude Linear Regression Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Northern Hemisphere\n",
    "x_values= northern_hemi_df[\"Lat\"]\n",
    "y_values= northern_hemi_df[\"Cloudiness\"]\n",
    "plot_linear_regression(x_values, y_values, \"Cloudiness\", (40, 20))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Southern Hemisphere\n",
    "x_values= southern_hemi_df[\"Lat\"]\n",
    "y_values= southern_hemi_df[\"Cloudiness\"]\n",
    "plot_linear_regression(x_values, y_values, \"Cloudiness\", (-50, 20))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Discussion about the linear relationship:** YOUR RESPONSE HERE\n",
    "The regression linear graphs show that cloudiness basically stays constant for the Southern Hemisphere. On the other hand, the Northern Hemisphere shows some decrease in cloudiness as latitude increases."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Wind Speed vs. Latitude Linear Regression Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Northern Hemisphere\n",
    "x_values= northern_hemi_df[\"Lat\"]\n",
    "y_values= northern_hemi_df[\"Wind Speed\"]\n",
    "plot_linear_regression(x_values, y_values, \"Wind Speed\", (10, 15))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear regression on Southern Hemisphere\n",
    "x_values= southern_hemi_df[\"Lat\"]\n",
    "y_values= southern_hemi_df[\"Wind Speed\"]\n",
    "plot_linear_regression(x_values, y_values,  \"Wind Speed\", (-50, 10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Discussion about the linear relationship:** YOUR RESPONSE HERE\n",
    "The regression linear graphs show that wind speed drops when latitude goes up for the Southern Hemisphere. The wind speed remains constant for the most part in the Northern Hemisphere."
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernel_info": {
   "name": "python3"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  },
  "latex_envs": {
   "LaTeX_envs_menu_present": true,
   "autoclose": false,
   "autocomplete": true,
   "bibliofile": "biblio.bib",
   "cite_by": "apalike",
   "current_citInitial": 1,
   "eqLabelWithNumbers": true,
   "eqNumInitial": 1,
   "hotkeys": {
    "equation": "Ctrl-E",
    "itemize": "Ctrl-I"
   },
   "labels_anchors": false,
   "latex_user_defs": false,
   "report_style_numbering": false,
   "user_envs_cfg": false
  },
  "nteract": {
   "version": "0.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}



Vacationpy code



{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# VacationPy\n",
    "---\n",
    "\n",
    "## Starter Code to Import Libraries and Load the Weather and Coordinates Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dependencies and Setup\n",
    "import hvplot.pandas\n",
    "import pandas as pd\n",
    "import requests\n",
    "\n",
    "#Import geoapify key\n",
    "from api_keys import geoapify_key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Load the CSV file created in Part 1 into a Pandas DataFrame\n",
    "city_data_df = pd.read_csv(\"output_data/cities.csv\")\n",
    "\n",
    "# Display the data\n",
    "city_data_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "### Step 1: Create a map that displays a point for every city in the `city_data_df` DataFrame. The size of the point should be the humidity in each city."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture --no-display\n",
    "\n",
    "#Make the map plot\n",
    "map_plot = city_data_df.hvplot.points(\n",
    "    \"Lng\",\n",
    "    \"Lat\",\n",
    "    geo=True,\n",
    "    size=\"Humidity\",\n",
    "    scale=1,\n",
    "    color=\"City\",\n",
    "    alpha=0.5,\n",
    "    tiles=\"OSM\",\n",
    "    frame_width=700,\n",
    "    frame_height=500\n",
    ")\n",
    "\n",
    "#Show the map\n",
    "map_plot"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Narrow down the `city_data_df` DataFrame to find your ideal weather condition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Narrow down cities that fit my own criteria\n",
    "narrowed_city_df = city_data_df[\n",
    "    (city_data_df[\"Max Temp\"] < 40) & (city_data_df[\"Max Temp\"] > 15) \\\n",
    "    & (city_data_df[\"Wind Speed\"] < 2.5) \\\n",
    "    & (city_data_df[\"Cloudiness\"] == 0)\n",
    "]  \n",
    "# Drop any rows with null values\n",
    "narrowed_city_df = narrowed_city_df.dropna()\n",
    "\n",
    "# Display sample data\n",
    "narrowed_city_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Create a new DataFrame called `hotel_df`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use the Pandas copy function to create DataFrame called hotel_df to store the city, country, coordinates, and humidity\n",
    "hotel_df = narrowed_city_df[[\"City\", \"Country\", \"Lat\", \"Lng\", \"Humidity\"]].copy()\n",
    "\n",
    "# Add an empty column, \"Hotel Name,\" to the DataFrame so you can store the hotel found using the Geoapify API\n",
    "hotel_df[\"Hotel Name\"]= \"\"\n",
    "\n",
    "# Display sample data\n",
    "hotel_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: For each city, use the Geoapify API to find the first hotel located within 10,000 metres of your coordinates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set parameters to search for a hotel\n",
    "radius = 10000\n",
    "params = {\n",
    "    \"categories\": \"accommodation.hotel\",\n",
    "    \"apiKey\": geoapify_key,\n",
    "    \"limit\": 20\n",
    "}\n",
    "\n",
    "# Print a message to follow up the hotel search\n",
    "print(\"Starting hotel search\")\n",
    "\n",
    "# Iterate through the hotel_df DataFrame\n",
    "for index, row in hotel_df.iterrows():\n",
    "    # get latitude, longitude from the DataFrame\n",
    "    latitude = row[\"Lat\"]\n",
    "    longitude = row[\"Lng\"]\n",
    "    \n",
    "    \n",
    "    # Add filter and bias parameters with the current city's latitude and longitude to the params dictionary\n",
    "    params[\"filter\"] = f\"circle:{longitude},{latitude},{radius}\"\n",
    "    params[\"bias\"] = f\"proximity:{longitude},{latitude}\"\n",
    "    \n",
    "    # Set base URL\n",
    "    base_url = \"https://api.geoapify.com/v2/places\"\n",
    "\n",
    "\n",
    "    # Make and API request using the params dictionaty\n",
    "    name_address = requests.get(base_url, params=params)\n",
    "    \n",
    "    # Convert the API response to JSON format\n",
    "    name_address = name_address.json()\n",
    "    \n",
    "    # Grab the first hotel from the results and store the name in the hotel_df DataFrame\n",
    "    try:\n",
    "        hotel_df.loc[index, \"Hotel Name\"] = name_address[\"features\"][0][\"properties\"][\"name\"]\n",
    "    except (KeyError, IndexError):\n",
    "        # If no hotel is found, set the hotel name as \"No hotel found\".\n",
    "        hotel_df.loc[index, \"Hotel Name\"] = \"No hotel found\"\n",
    "        \n",
    "    # Log the search results\n",
    "    print(f\"{hotel_df.loc[index, 'City']} - nearest hotel: {hotel_df.loc[index, 'Hotel Name']}\")\n",
    "\n",
    "# Display sample data\n",
    "hotel_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Add the hotel name and the country as additional information in the hover message for each city in the map."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture --no-display\n",
    "\n",
    "#Configure the map plot\n",
    "map_plot= hotel_df.hvplot.points(\n",
    "    \"Lng\",\n",
    "    \"Lat\",\n",
    "    geo=True,\n",
    "    size=\"Humidity\",\n",
    "    scale=1,\n",
    "    color=\"City\",\n",
    "    alpha=0.5,\n",
    "    tiles=\"OSM\",\n",
    "    frame_width=700,\n",
    "    frame_height=500,\n",
    "    hover_cols=[\"Hotel Name\", \"Country\"]\n",
    ")\n",
    "\n",
    "# Display the map\n",
    "map_plot"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  },
  "latex_envs": {
   "LaTeX_envs_menu_present": true,
   "autoclose": false,
   "autocomplete": true,
   "bibliofile": "biblio.bib",
   "cite_by": "apalike",
   "current_citInitial": 1,
   "eqLabelWithNumbers": true,
   "eqNumInitial": 1,
   "hotkeys": {
    "equation": "Ctrl-E",
    "itemize": "Ctrl-I"
   },
   "labels_anchors": false,
   "latex_user_defs": false,
   "report_style_numbering": false,
   "user_envs_cfg": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
