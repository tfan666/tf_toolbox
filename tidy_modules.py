import numpy as np
import pandas as pd
from google.cloud import bigquery, storage


def check_bq_price(sql):
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query_job = client.query(sql, job_config=job_config)  # Make an API request.
    QUERY_COST = round(query_job.total_bytes_processed*1e-12*5, 3)
    # A dry run query completes immediately.
    print(f"This query will cost $ {QUERY_COST}.")

def read_bq(sql, dry_run=False):
    if dry_run==False:
        client = bigquery.Client()
        df = client.query(sql).to_dataframe()
        return df    
    else:
        check_bq_price(sql=sql)
