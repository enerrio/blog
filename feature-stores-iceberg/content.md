# Exploring Feature Stores with Apache Iceberg

What is a feature store and why do you need one? Imagine you work as an engineer for the Bank of Saturn and your job is to help keep customers safe from fraud. Every day, transaction data from customer purchases comes through your servers and into your data warehouse. Your team crafts features from those purchases and uses them as training data to build an anomaly detection system. This is a situation where you would use a feature store! But why exactly would you need one? Think of it as a centralized place to store everything you would want to know about your anomaly detection system, from the model version number to the exact snapshot of the training data used. This is extremely useful for keeping track of all your features and model metadata, as well as for auditing and debugging purposes which we'll see later.

The term feature store is kinda generic and different companies[^1] have different[^2] definitions[^3] but in short: A feature store is a database that contains data relevant to the training of predictive models. There may be different use cases or capabilities available in different implementations. Some might store only features used for training a certain type of model (like anomaly detectors) while others might be more like a general catalog that stores features from disparate sources not tied to any one specific use case.

In this post we'll walk through the use case of a feature store from the POV of that engineer helping build an anomaly detector. We'll store all the data in our feature store: raw transactions, features, and model metadata. We'll also give our feature store the capability to do time travel querying by utilizing Apache Iceberg. This can be accomplished by taking "snapshots" of the data every time new data is ingested and allows you to query the data at different points in time. We'll see later how this can be used to investigate a model by querying the exact historical data it was trained on, even if new data has already been ingested.

If you want to follow along, check out the [Github repo](https://github.com/enerrio/feature-store-iceberg) for this post.

## Crash Course in Apache Iceberg
We're going to be using [Apache Iceberg](https://iceberg.apache.org) to organize our data into tables. Iceberg is a popular open source table format that organizes data tables and has some useful features like schema evolution, hidden partitioning, and time travel. I chose it for this project because it's time travel capabilities will help demonstrate the usefulness of keeping "snapshots" of data that can be used by a feature store and for the debugging part of this project. Iceberg is similar to [delta lake](https://delta.io) which has similar features but is geared more towards Spark, while Iceberg is more flexible. Check out this [article](https://www.starburst.io/blog/iceberg-vs-delta-lake/) for a pragmatic comparison.

When getting started with Iceberg you first have to set up a catalog. Catalogs are "managers" responsible for keeping track of the data tables's metadata. The tables themselves are further grouped together into "namespaces." Iceberg offers different catalog types like Nessie and Hive but we'll use their newer REST catalog which was introduced in 2022, mostly because I don't want to deal with Java jars 😭.

Once the catalog is created then you can create a namespace and start creating tables inside it. Creating a namespace is a very simple and necessary step before creating your tables. Without a namespace, the catalog won't be able to find and track your tables! When you create your tables you first define a schema which can optionally include partitions. Unlike other relational databases where your partition is just another column, Iceberg uses hidden partitioning which means that your partition is derived from an existing column in your dataset and then "hidden" from the user. For example if you have a "timestamp" column (e.g. 2025-09-14 12:30:00) and you wanted to partition on the day part of that timestamp, then you could create a partition called "dt" that extracts the date part of the timestamp column (e.g. 2025-09-14). That new "dt" partition isn't visible when you query the data but if you apply a filter on the timestamp column, then Iceberg will use the partition under the hood to make your query run faster.

```sql
-- This query uses the `dt` partition under the hood to make it more efficient
SELECT
    *
FROM icecat.default.raw_events
WHERE
    event_timestamp BETWEEN '2025-08-15' AND '2025-09-10'
```

Once the table is created then we can start ingesting data. In Iceberg, the terminology is "appending" data to a table. Every time you append new data to a table, a snapshot is created and identified with a unique ID. This will be useful later for time travel querying.

We can query the data stored in these tables by using a query engine like DuckDB, or any other engine you like.

That's enough about Iceberg (for now). Check out the PyIceberg (Python API for Iceberg) [documentation](https://py.iceberg.apache.org) for more info!

## Crafting the Data
To get started with a feature store, we first need data! The [`data_generator.py`](https://github.com/enerrio/feature-store-iceberg/blob/main/src/data_generator.py) script will handle creating random data along a lognormal distribution to simulate user spending. We'll also have functionality to optionally inject anomalies into the data. Since this is the first step of our pipeline, we'll also set up the catalog and a namespace called "default." Before ingesting the generated data we need to first create a table. Let's call it `raw_events`:

| Column Name     | Data Type      | Description                          |
| --------------- | -------------- | ------------------------------------ |
| id              | BIGINT         | Unique identifier for the event      |
| user_id         | BIGINT         | ID of the user associated with event |
| amount          | DECIMAL(15, 2) | Dollar amount spent                  |
| vendor_id       | BIGINT         | Vendor where transaction occurred    |
| event_timestamp | TIMESTAMP      | When the event occurred in UTC       |

A hidden partition named "dt" is also created and derived from the `event_timestamp` column for efficient querying across time.

If we wanted to query this data we could set up a DuckDB connection from our local machine to an Iceberg server running inside Docker like so:
```python
import duckdb

# Set up connection
conn = duckdb.connect()
conn.sql("""
INSTALL iceberg;
LOAD iceberg;
UPDATE EXTENSIONS;

ATTACH 'warehouse' AS icecat (TYPE ICEBERG, ENDPOINT 'http://localhost:8181', AUTHORIZATION_TYPE 'none');

USE icecat.default;
""")

# Query data
conn.sql("SELECT COUNT(*) FROM icecat.default.raw_events;")
```

Here is the [`docker-compose.yml`](https://github.com/enerrio/feature-store-iceberg/blob/main/docker-compose.yml) file that sets up Iceberg in it's own docker container. It exposes port 8181 so we can connect to it from outside the container like we did above with DuckDB.

For more information about the Iceberg extension for DuckDB, check out the documentation [here](https://duckdb.org/docs/stable/core_extensions/iceberg/overview.html).

Now that we have our `raw_events` table ready to go, we can transform it into features used to train our anomaly detector.

## Feature Engineering
We can build a mini-ETL pipeline to extract the data from `raw_events`, transform them into model-ready features, and load them into a new table called `user_features`. To keep things simple we'll have a single feature available: The mean spending amount for each user across a 7-day rolling window. Days where any given user doesn't spend any money will count as $0 for that day, so the "gaps" are "filled" (days without any spending are called "gaps"). To make this clear, let's look at a quick example for a single user. The following table demonstrates the spending habits for the user Aaron across a week and calculates the 7 day mean:

| Day         | Amount Spent | Calculation                  | 7 Day Gap-Filled Mean |
| ----------- | ------------ | ---------------------------- | --------------------- |
| Sunday      | $7           | $7 ÷ 7                       | $1                    |
| Monday      | $7           | ($7 +$7) ÷ 7                 | $2                    |
| Tuesday     | $0           | ($7 +$7 + $0) ÷ 7            | $2                    |
| Wednesday   | $0           | ($7 +$7 + $0 + ...) ÷ 7      | $2                    |
| Thursday    | $0           | ($7 +$7 + $0 + ...) ÷ 7      | $2                    |
| Friday      | $4           | ($7 +$7 + $4 + $0 + ...) ÷ 7 | $2.57                 |
| Saturday    | $0           | ($7 +$7 + $4 + $0 + ...) ÷ 7 | $2.57                 |
| Next Sunday | $0           | ($7 + $4 + $0 + ...) ÷ 7     | $1.57                 |

Our `user_features` table schema will look like this:

| Column Name      | Data Type      | Description                               |
| ---------------- | -------------- | ----------------------------------------- |
| user_id          | BIGINT         | ID of the user associated with event      |
| dt               | DATE           | Date of last day in 7 day time period     |
| spending_mean_7d | DECIMAL(15, 2) | Average spending over a 7 day time period |

Our actual mini-ETL pipeline is simply running a SQL query to compute that calculation and then we ingest the resulting data into the new features table. The full query can be found in [`sql/gapfilled_7day_spend.sql`](https://github.com/enerrio/feature-store-iceberg/blob/main/sql/gapfilled_7day_spend.sql) and the whole ingestion code is in [`feature_engineer.py`](https://github.com/enerrio/feature-store-iceberg/blob/main/src/feature_engineer.py) which connects to the catalog, runs that SQL query, creates the new `user_features` table, and appends newly created data to it.

We're almost ready to set up our features store. We have all the key pieces for tracking our training pipeline except for the model metadata. But for that we first need a model! Let's create that next.

## Interlude: Building an Anomaly Detector
Our anomaly detector will be a simple model based on statistical heuristics. We'll calculate a global z-score based on a quantile and use that as a threshold for determining if a transaction is anomalous or not. So let's say that we choose a quantile value of 0.95. That means we'll look at the `spending_mean_7d` data we have and calculate z-scores for all data points. Then we'll look at the distribution of those scores and choose a cutoff based on our chosen quantile: 0.95. Any `spending_mean_7d` value beyond that is classified as an anomaly. In other words: we'll look at the data distribution for the `spending_mean_7d` column and anything **above** the 95th percentile (i.e. the far outliers) are anomalies.

This isn't a perfect detector. We're letting the spending habits of users affect each other. But it is simple. And for this post that's enough. We want to learn about feature stores, not anomaly detectors!

This simplicity is also the reason why I chose not to do an ML-based detector. An ML-based one would be a better fit for a feature store, where they are used primarily for machine learning use cases, but its inclusion may overshadow the feature store.

## Time Travel
Earlier I mentioned that Iceberg has built-in time travel querying. Let's explore that feature a bit. Say you ingest data into your `raw_events` table multiple times a day. Each append operation has a snapshot associated with it. Let's reuse the DuckDB connection that we set up earlier. You can query the history of your append operations:

```sql
SELECT * FROM iceberg_snapshots('icecat.default.raw_events');
```

In a situation where we've appended data four separate times, the result of the above query looks something like this:

| sequence_number | snapshot_id         | timestamp_ms               | manifest_list                                                                                                          |
| --------------- | ------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 4               | 5684812838621277724 | 2025-09-15 00:56:10.863000 | file:///tmp/warehouse/default/raw_events/metadata/snap-5684812838621277724-0-2ce3dcd1-4895-43d1-9b44-71918f5ab90b.avro |
| 3               | 8216567921051981541 | 2025-09-15 00:56:10.808000 | file:///tmp/warehouse/default/raw_events/metadata/snap-8216567921051981541-0-10cc8b5e-e05a-4171-b77a-a3434b92d169.avro |
| 2               | 6768109534693656825 | 2025-09-14 23:15:25.586000 | file:///tmp/warehouse/default/raw_events/metadata/snap-6768109534693656825-0-dd8391f8-e90e-4aca-83cb-fec1fd31f0b0.avro |
| 1               | 7252827444344540085 | 2025-09-14 23:15:25.533000 | file:///tmp/warehouse/default/raw_events/metadata/snap-7252827444344540085-0-1867ca2e-8331-4f43-9767-72406f11baa2.avro |

By querying a specific snapshot ID we can essentially query the data as it existed in that point in time! If we query just the snapshot associated with sequence ID 1 then we're looking at the data that just contains the first batch of data ingested i.e. the 1st append operation. And if we query the snapshot associated with sequence ID 2 then we're looking at the data for the first two append operations.

```sql
-- Time travel!
SELECT
    COUNT(*)
FROM iceberg_scan("icecat.default.raw_events", snapshot_from_id=6768109534693656825)
```

We'll use this capability inside our feature store to retrieve the training data associated with a specific training run and also when we need to debug a model's behavior.

## Feature Store
Finally! Our feature store will be responsible for creating a new table that tracks our anomaly detection model's metadata, including the training data used to create it. Let's take a look at the schema for this table:

| Column Name            | Data Type | Description                                                 |
| ---------------------- | --------- | ----------------------------------------------------------- |
| model_version          | VARCHAR   | Model version number                                        |
| trained_at             | TIMESTAMP | Time that model completed training                          |
| feature_snapshot_id    | BIGINT    | Snapshot ID of `user_features` used for training            |
| raw_events_snapshot_id | BIGINT    | Snapshot ID of `raw_events` used for creating features      |
| feature_name           | VARCHAR   | Feature column used when calibrating the threshold          |
| decision_threshold     | DOUBLE    | Final \|z\|-score threshold chosen from the training window |
| training_window_start  | DATE      | First day of the training window                            |
| training_window_end    | DATE      | Last day of the training window                             |
| quantile               | DOUBLE    | Quantile used to pick the threshold (optional)              |

![meme](https://i.ibb.co/WWTSZWYp/featstore.jpg)

Yes, our feature store is essentially just another table in our Iceberg catalog! The key columns being stored here are the snapshot IDs for the `user_features` and `raw_events` tables plus the calibrated threshold for the feature column that powered the detector. One row contains enough information to time-travel back to the raw data used to create the model, reproduce the feature set, and explain exactly why a detection fired.

We'll also have some helper methods that can retrieve the actual data for a model version. Take a look at the full implementation in [`feature_store.py`](https://github.com/enerrio/feature-store-iceberg/blob/main/src/feature_store.py).

## Running the Detector
If you want to try this whole pipeline out yourself, check out the repo that has all the code. There is a [`Makefile`](https://github.com/enerrio/feature-store-iceberg/blob/main/Makefile) included that has shortcuts for walking through the detector workflow without writing long bash commands by hand. To get started make sure your Iceberg is running inside a docker container and then you can run the `Makefile` targets.

```bash
# Launch Iceberg in a docker container. Detach so you can run other commands
docker compose up -d

# Generate fresh raw events
make data

# Generate fresh raw events with some anomalies injected for a single user
make data ANOM_FILE=anomaly_files/single_user.json

# Create the user feature table and populate with transformed raw events
make features

# Calibrate and register a model. Store metadata table inside "default" namespace
make train-model NAMESPACE=default

# Make a prediction for a specific transaction amount (model version is from `make train-model` output)
make detect AMOUNT=50 MODEL_VERSION=v20250916224417

# Explain a specific transaction
make debug USER_ID=42 AMOUNT=275 MODEL_VERSION=v20250916224417

# Tear down the docker container when you're done. Also remove any attached volumes
docker compose down -v
```

## Debugging a Model with our Feature Store
The next step we can take is to examine how we can debug our detector. Let's say that our model is deployed and all of the sudden customers start calling into the Bank of Saturn complaining that their credit cards are being frozen because too many of their transactions are being flagged as anomalous.

And let's also say that we have different models deployed to different customer segments. In this case we can take those anomalous transactions and debug it by running inference and retrieving the z-score and threshold calculated when the model was first "trained" (more like calibrated since we're not doing any machine learning here). We can also get the some historical values that were used at the time of training with the snapshot IDs for the `user_features` and `raw_events` tables. This can be helpful context when figuring out why a specific transaction was flagged, but we can take it a step further even. We can investigate the exact data used to train the model using the snapshot ID to time travel. We can even "branch off" by editing the feature's snapshot and retraining a new version of the model.

In our `Makefile`, the `make detect` target mirrors the debug target but only returns the scoring decision. Meanwhile `make debug` will return a scoring decision **and** give an explanation for the reasoning behind the decision described above.

Here's an example output for one model `make debug USER_ID=42 AMOUNT=275 MODEL_VERSION=v20250916224417`:
```json
{
  "model_version": "v20250916224417",
  "decision": {
    "is_anomaly": true,
    "z_score": 8.587255756547487,
    "historical_mean": 42.210361445783136,
    "historical_std": 27.10873475227785,
    "threshold_used": 3.892545700073242
  },
  "why": "|z-score| 8.59 exceeds threshold 3.89",
  "context": {
    "as_of": "2025-09-17",
    "user_id": 42,
    "as_of_feature_value": 18.49,
    "last_n_feature_values": [
      15.02,
      30.56,
      30.56,
      26.59,
      26.59
    ],
    "last_n_feature_dates": [
      "2025-09-02",
      "2025-09-03",
      "2025-09-04",
      "2025-09-05",
      "2025-09-06"
    ]
  },
  "model_card": {
    "trained_at": "2025-09-17 05:44:17.854480+00:00",
    "feature_snapshot_id": 4199953139027461656,
    "raw_events_snapshot_id": 2284627372037259730,
    "feature_name": "spending_mean_7d",
    "decision_threshold": 3.892545700073242,
    "training_window_start": "2025-08-18",
    "training_window_end": "2025-09-17",
    "quantile": 0.9950000047683716
  }
}
```

This kind of information can be useful for auditing use cases as well.

## Closing Thoughts
Now that you have an idea of what feature stores are, you can understand why they are commonly used in production systems. Their purpose extends beyond helping teams better organize their data and model metadata, but can also be used for auditing and debugging purposes.

There's much more to explore with feature stores. Now that you understand the fundamentals, you can explore how real world implementations like [Feast](https://feast.dev) work and see all the extra benefits they provide.

In this post we've walked through building a fully functional (albeit simple) feature store and shown how it can help with tracking past training runs. Also learned a little bit about Apache Iceberg and how it is a super useful tool for keeping track of your data tables.

## References
[^1]: Tecton - [What is a Feature Store?](https://www.tecton.ai/blog/what-is-a-feature-store/)
[^2]: Databricks - [Feature Store Documentation](https://docs.databricks.com/aws/en/machine-learning/feature-store/)
[^3]: Feature Form - [Feature Store Explained](https://www.featureform.com/post/feature-stores-explained-the-three-common-architectures)
