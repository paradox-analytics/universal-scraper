# Deployment Fix Applied

## Issue Found
The deployment failed with error:
```
ERROR: spec.template.spec.containers[0].env: The following reserved env names were provided: PORT. These values are automatically set by the system.
```

## Fix Applied
Removed the `--set-env-vars PORT=8080` from `cloudbuild.yaml` because Cloud Run automatically sets the PORT environment variable.

## New Deployment
A new deployment has been started. This should complete successfully in ~10-15 minutes.

## Status
Check deployment status:
```bash
gcloud builds list --limit=1 --project=soma-data-467016
```

Once deployment completes, the document processing endpoint will be available.




