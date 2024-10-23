import dagshub
import mlflow

dagshub.init(repo_owner='shishant1984', repo_name='MLops_project', mlflow=True )

mlflow.set_tracking_uri('https://dagshub.com/shishant1984/MLops_project.mlflow')
