# legisplain


Tools to bulk download and process legislative XML.


# Data Refresh Playbook


## Clone the repo

```bash
git clone git@github.com:galtay/legisplain.git
```

## Update main config

Copy the template config file

```bash
cp config_template.yml config.yml
```

Update the variables in `config.yml`
For example,

```yaml
---
bulk_path: "/home/user/data/congress-bulk"
hf_path: "/home/user/data/congress-hf"
s3_bucket: "hyperdemocracy"
pg_conn_str: "postgresql+psycopg2://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
```

## Write child scripts

```bash
uv run main.py write-scripts
```

## Run scripts

```bash
cd scripts/bulk_download
./download-and-sync-with-log.sh
```

## Populate postgres

```bash
uv run main.py pg-populate-billstatus
uv run main.py pg-populate-textversion
uv run main.py pg-populate-unified
```


## Upload to Hugging Face

```bash
uv run main.py hf-upload-billstatus
uv run main.py hf-upload-textversion
uv run main.py hf-upload-unified
```
