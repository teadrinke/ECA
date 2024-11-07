### How to run the ECA application

1. Postgres Setup
Before starting the ECA application, ensure that Postgres is installed on your system. Follow these steps to set up the Postgres database:

1. Postgres Installation:
   Download and install Postgres from the official website for your operating system.

2. Start Postgres Server:
   Open Postgres Workbench and start the server.

3. Connect to Postgres:
   Open a terminal window and enter the following command to connect to the Postgres database:
   psql postgres <username>

4. Create Database:
   Create a new database named `eca` by executing the following SQL command:
   CREATE DATABASE eca;
   

5. Switch to Database:
   Connect to the `eca` database:
  
   \c eca
   

6. Create Table:
   Create a table named `Transcripts` with the required schema:
   CREATE TABLE Transcripts (
       id SERIAL PRIMARY KEY,
       Ticker TEXT,
       Year INT,
       Quarter INT,
       Speaker TEXT,
       Designation TEXT,
       Paragraph TEXT,
       Processed_Response TEXT,
       Company TEXT
   );
   

7. Copy Data from CSV:
   Copy the data from the provided CSV file (`Seven_companies_with_company_name.csv`) into the `Transcripts` table:
   COPY Transcripts(Ticker, Year, Quarter, Speaker, Designation, Paragraph, Processed_Response, Company)
   FROM 'Seven_companies_with_company_name.csv'
   DELIMITER ','
   CSV HEADER;


2. Elasticsearch Setup
Follow these steps to set up Elasticsearch and load data into it:

1. Elasticsearch setup: https://dylancastillo.co/elasticsearch-python/ 
   

2. Load Data into Elasticsearch:
   Run the provided Python script `loader.py` to load data into Elasticsearch. Make sure to update the script with the necessary configuration details.

3. Environment Variables
Create a `.env` file in the root directory of the project with the following variables:


DB_HOST=localhost
DB_PORT=5432
DB_NAME=eca
DB_USER=<your_username>
DB_PASSWORD=<your_password>
ES_PORT=9200


4. Running the Application
Run the Flask application by executing the following command in the terminal:

python3 app.py


### Additional Notes
- Make sure to replace `<your_username>` and `<your_password>` in the `.env` file with your actual Postgres credentials.
- Ensure that all dependencies are installed by running `pip install -r requirements.txt` before starting the application.

This technical documentation provides detailed instructions for setting up and running the ECA application, including Postgres and Elasticsearch configuration, environment variable setup, and running the Flask application.




